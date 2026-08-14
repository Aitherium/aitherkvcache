"""RoPE strip / re-apply — the step that makes a cross-model KV map fittable at all.

## Why this module exists

A cross-model KV mapper is a **position-independent** linear map: one matrix per
(target layer, target head), applied identically to every token. Keys as they
appear in a live KV cache are not position-independent — RoPE has already
rotated each key by an angle proportional to its position. The same content at
position 10 and position 4000 is a *different vector*, related by a rotation
that the map is not allowed to know about.

So a map fitted on rotated keys is being asked to learn "the source model's
representation, composed with an arbitrary rotation that varies per row". It
cannot: the best a single matrix can do is average over the rotations seen in
training, which drives R^2 toward the position-marginal and produces a mapper
that looks trained, loads cleanly, and reconstructs noise.

The fix is the one the paper describes: strip the SOURCE model's rotation, fit
in position-free space, re-apply the TARGET model's rotation at inference. That
last asymmetry matters — the two models may disagree on ``rope_theta`` (SmolLM2
135M uses 100000, the 1.7B uses 130000), so "strip and re-apply" is not a no-op
even when the head dimension matches.

**Values are never rotated.** The V path skips this module entirely. Passing V
through it is not a small error, it is a corruption, so ``strip`` takes the
tensor role explicitly rather than inferring it.

## The convention is load-bearing, not a detail

HuggingFace llama/qwen RoPE pairs dimension ``i`` with ``i + d/2`` (the
``rotate_half`` convention), NOT adjacent pairs ``(2i, 2i+1)`` as in the
original RoPE paper and in GGML. Both are "RoPE". Choosing wrong yields a
rotation that is still orthogonal and still invertible, so every round-trip test
passes and every shape check passes — and the fitted map is garbage, because
position-free space was never reached. ``_ROTATE_HALF`` is asserted by
``--self-test`` against a hand-computed 2D rotation, and the mutation guard
there reproduces the interleaved convention to prove the test can tell them
apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Rope scalings we reproduce exactly. Anything else (llama3, longrope, dynamic)
# changes the frequency schedule, and computing the wrong frequencies is silent:
# the strip runs, the numbers look plausible, and the fit is against the wrong
# basis. Refuse instead of guessing.
_SUPPORTED_ROPE_TYPES = frozenset({"", "default", "linear", "yarn"})


@dataclass(frozen=True)
class RopeSpec:
    """Everything needed to reproduce one model's key rotation.

    ``head_dim`` is here rather than taken from the tensor because a mismatch
    between the config and the captured tensor is itself a defect worth failing
    on — it means the capture came from a different model than the manifest says.
    """

    head_dim: int
    theta: float
    rope_type: str = "default"
    scaling_factor: float = 1.0
    # YaRN only. Defaults are the DeepSeek/HF reference values.
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_max_position_embeddings: int = 4096
    mscale: float = 1.0

    def __post_init__(self) -> None:
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be positive and even, got {self.head_dim}")
        if not (self.theta > 0):
            raise ValueError(f"rope theta must be positive, got {self.theta}")
        rt = (self.rope_type or "default").lower()
        if rt not in _SUPPORTED_ROPE_TYPES:
            raise ValueError(
                f"unsupported rope_type {self.rope_type!r}: this module reproduces only "
                f"{sorted(_SUPPORTED_ROPE_TYPES)}. A scaled/dynamic schedule must be "
                "implemented explicitly — approximating it corrupts the fit silently."
            )
        if not (self.scaling_factor > 0):
            raise ValueError(f"scaling_factor must be positive, got {self.scaling_factor}")
        if rt == "yarn" and abs(self.mscale - 1.0) > 1e-9:
            # DeepSeek's YaRN multiplies BOTH cos and sin by an attention scale
            # m. That makes the transform a scaled rotation, so inverting it is
            # not "negate sin" — that would come back m^2 too large. Every model
            # we need today has m = 1; rather than ship an inverse that is
            # silently wrong by a constant factor for the ones that do not,
            # refuse until it is implemented and tested against a reference.
            raise ValueError(
                f"yarn mscale={self.mscale} is not 1.0: cos/sin are scaled, so the "
                "inverse is not sin-negation. Implement and test the scaled inverse "
                "before enabling this model."
            )

    @classmethod
    def from_hf_config(cls, config: object) -> "RopeSpec":
        """Build from a transformers config, refusing anything we cannot reproduce."""
        head_dim = getattr(config, "head_dim", None)
        if not head_dim:
            hidden = int(getattr(config, "hidden_size"))
            n_heads = int(getattr(config, "num_attention_heads"))
            head_dim = hidden // n_heads
        scaling = getattr(config, "rope_scaling", None) or {}
        rope_type = str(scaling.get("rope_type") or scaling.get("type") or "default")
        factor = float(scaling.get("factor", 1.0) or 1.0)
        if rope_type.lower() == "yarn":
            return cls(
                head_dim=int(head_dim),
                theta=float(getattr(config, "rope_theta", 10000.0)),
                rope_type=rope_type,
                scaling_factor=factor,
                beta_fast=float(scaling.get("beta_fast", 32.0)),
                beta_slow=float(scaling.get("beta_slow", 1.0)),
                original_max_position_embeddings=int(
                    scaling.get("original_max_position_embeddings")
                    or getattr(config, "max_position_embeddings", 4096)),
                mscale=float(scaling.get("mscale", 1.0) or 1.0),
            )
        return cls(
            head_dim=int(head_dim),
            theta=float(getattr(config, "rope_theta", 10000.0)),
            rope_type=rope_type,
            scaling_factor=factor,
        )

    def _yarn_correction_range(self) -> tuple[float, float]:
        """Dimension indices between which YaRN ramps interpolation -> extrapolation.

        Reproduces DeepSeek's ``yarn_find_correction_dim`` / ``_range``. The
        wavelength of pair i is 2*pi*theta^(2i/d); YaRN extrapolates the pairs
        whose wavelength is short relative to the original training context and
        interpolates the long ones, blending across the middle.
        """
        def dim_for(rotations: float) -> float:
            return (self.head_dim * math.log(
                self.original_max_position_embeddings / (rotations * 2 * math.pi))
            ) / (2 * math.log(self.theta))

        low = math.floor(dim_for(self.beta_fast))
        high = math.ceil(dim_for(self.beta_slow))
        return max(low, 0.0), min(high, self.head_dim - 1.0)

    def inv_freq(self) -> np.ndarray:
        """Per-pair angular frequencies, shape (head_dim/2,)."""
        half = self.head_dim // 2
        exponents = np.arange(0, half, dtype=np.float64) * 2.0 / self.head_dim
        freqs = 1.0 / (self.theta ** exponents)
        rt = (self.rope_type or "default").lower()
        if rt == "linear":
            # "linear" scaling divides POSITIONS by factor, which is identical to
            # dividing the frequencies — done here so callers stay position-honest.
            freqs = freqs / self.scaling_factor
        elif rt == "yarn":
            # NTK-by-parts: blend extrapolated (unscaled) and interpolated
            # (factor-scaled) frequencies with a linear ramp over the correction
            # range. Scaling every frequency uniformly — the naive reading of
            # "factor" — destroys the high-frequency pairs that encode local
            # order, which is precisely what YaRN exists to avoid.
            freq_extra = freqs
            freq_inter = freqs / self.scaling_factor
            low, high = self._yarn_correction_range()
            if high - low < 1e-3:
                high = low + 0.001
            ramp = (np.arange(half, dtype=np.float64) - low) / (high - low)
            ramp = np.clip(ramp, 0.0, 1.0)
            extrapolation_mask = 1.0 - ramp
            freqs = freq_inter * (1.0 - extrapolation_mask) \
                + freq_extra * extrapolation_mask
        return freqs

    def cos_sin(self, positions: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """cos/sin tables of shape (n_positions, head_dim), duplicated halves.

        The duplication (``concat(freqs, freqs)``) is what makes the rotate_half
        convention work: dimension i and i+d/2 share one angle.
        """
        pos = np.asarray(positions, dtype=np.float64).reshape(-1)
        angles = pos[:, None] * self.inv_freq()[None, :]
        emb = np.concatenate([angles, angles], axis=-1)
        return np.cos(emb), np.sin(emb)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    """HF convention: pair dim i with i+d/2. See the module docstring."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(
    keys: np.ndarray,
    spec: RopeSpec,
    positions: Sequence[int] | np.ndarray,
    *,
    inverse: bool = False,
) -> np.ndarray:
    """Rotate (or un-rotate) keys in place-free fashion.

    Args:
        keys: (..., n_positions, head_dim). The position axis is second-to-last.
        spec: the rotation schedule to use — SOURCE's when stripping, TARGET's
            when re-applying. These are deliberately separate calls.
        positions: absolute position of each row. Not ``range(len)`` by default:
            a translated cache is frequently a *suffix* of a conversation, and
            assuming it starts at 0 rotates every key by the wrong angle while
            failing nothing.
        inverse: negate the angle, i.e. strip an existing rotation.

    Returns:
        float32 array of the same shape.
    """
    if keys.shape[-1] != spec.head_dim:
        raise ValueError(
            f"key head_dim {keys.shape[-1]} != RopeSpec.head_dim {spec.head_dim} — "
            "the capture and the config disagree about which model this is"
        )
    n_pos = keys.shape[-2]
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    if pos.shape[0] != n_pos:
        raise ValueError(f"got {pos.shape[0]} positions for {n_pos} key rows")

    cos, sin = spec.cos_sin(pos)
    if inverse:
        sin = -sin
    # Broadcast (n_pos, d) against (..., n_pos, d).
    x = keys.astype(np.float64, copy=False)
    out = x * cos + _rotate_half(x) * sin
    return out.astype(np.float32, copy=False)


def strip_rope(
    keys: np.ndarray,
    spec: RopeSpec,
    positions: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Remove the source model's rotation, landing keys in position-free space."""
    return apply_rope(keys, spec, positions, inverse=True)


def rotate_to_target(
    keys_unrotated: np.ndarray,
    target_spec: RopeSpec,
    positions: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Apply the TARGET model's rotation to a position-free key block."""
    return apply_rope(keys_unrotated, target_spec, positions, inverse=False)


# ── self-test ───────────────────────────────────────────────────────────────


def _self_test() -> int:
    """Prove the rules can still fail. Exit 0 pass, 1 fail."""
    rng = np.random.default_rng(20260813)
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(f"{name}: {detail}")

    spec = RopeSpec(head_dim=8, theta=10000.0)
    positions = np.arange(5)
    keys = rng.standard_normal((2, 3, 5, 8)).astype(np.float32)

    # 1. Round trip is the identity.
    back = strip_rope(apply_rope(keys, spec, positions), spec, positions)
    check("roundtrip", float(np.abs(back - keys).max()) < 1e-4,
          f"max abs err {float(np.abs(back - keys).max()):.2e}")

    # 2. Position 0 is a no-op (angle 0) — catches an off-by-one in positions.
    at_zero = apply_rope(keys[..., :1, :], spec, [0])
    check("pos0_identity", float(np.abs(at_zero - keys[..., :1, :]).max()) < 1e-5)

    # 3. The rotation is NOT the identity at nonzero position. Without this, a
    #    stubbed-out apply_rope passes checks 1 and 2 perfectly.
    at_one = apply_rope(keys[..., 1:2, :], spec, [1])
    check("pos1_moves", float(np.abs(at_one - keys[..., 1:2, :]).max()) > 1e-3,
          "rotation at position 1 did nothing — is apply_rope a no-op?")

    # 4. Norm preservation: a rotation is orthogonal.
    n_before = np.linalg.norm(keys, axis=-1)
    n_after = np.linalg.norm(apply_rope(keys, spec, positions), axis=-1)
    check("orthogonal", float(np.abs(n_before - n_after).max()) < 1e-3)

    # 5. The convention itself, against hand arithmetic. d=4 pairs (0,2) and
    #    (1,3); at position 1 the first pair rotates by exactly 1 radian.
    s2 = RopeSpec(head_dim=4, theta=10000.0)
    unit = np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    got = apply_rope(unit, s2, [1])[0, 0]
    want_x, want_y = math.cos(1.0), math.sin(1.0)
    check("rotate_half_convention",
          abs(got[0] - want_x) < 1e-5 and abs(got[2] - want_y) < 1e-5,
          f"expected dims (0,2) = ({want_x:.4f},{want_y:.4f}), got "
          f"({got[0]:.4f},{got[2]:.4f}) — wrong pairing convention?")

    # 5b. MUTATION GUARD for check 5: the interleaved (2i, 2i+1) convention is
    #     also orthogonal and also round-trips, so checks 1-4 cannot separate
    #     them. Prove check 5 can.
    def _rotate_interleaved(x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[..., 0::2] = -x[..., 1::2]
        y[..., 1::2] = x[..., 0::2]
        return y

    cos_t, sin_t = s2.cos_sin([1])
    # Interleaved variant needs interleaved angle duplication to be a fair rival.
    ang = np.repeat(s2.inv_freq() * 1.0, 2)[None, :]
    wrong = unit * np.cos(ang) + _rotate_interleaved(unit) * np.sin(ang)
    check("mutation_guard_convention",
          abs(wrong[0, 0, 2] - want_y) > 1e-3,
          "the interleaved convention produced the SAME answer as rotate_half — "
          "check 5 cannot distinguish conventions and is not a real assertion")
    del cos_t, sin_t

    # 6. Differing theta really changes the result: strip-with-source then
    #    apply-with-target must NOT be an identity. This is the whole reason
    #    re-application is a separate call.
    src = RopeSpec(head_dim=8, theta=100000.0)
    tgt = RopeSpec(head_dim=8, theta=130000.0)
    moved = rotate_to_target(strip_rope(keys, src, positions), tgt, positions)
    check("theta_asymmetry", float(np.abs(moved - keys).max()) > 1e-4,
          "different rope_theta produced identical keys — re-application is inert")

    # 7. YaRN, against the DeepSeek-V4-Flash schedule.
    yarn = RopeSpec(head_dim=64, theta=10000.0, rope_type="yarn", scaling_factor=16.0,
                    beta_fast=32.0, beta_slow=1.0,
                    original_max_position_embeddings=65536)
    plain = RopeSpec(head_dim=64, theta=10000.0)
    yf, pf = yarn.inv_freq(), plain.inv_freq()

    check("yarn_differs", float(np.abs(yf - pf).max()) > 1e-9,
          "yarn frequencies are identical to default — the schedule is inert")

    # The defining property of NTK-by-parts: the HIGHEST-frequency pairs are
    # EXTRAPOLATED (left alone) and the lowest are INTERPOLATED (divided by
    # factor). A uniform division — the naive reading of "factor" — would fail
    # the first of these, and it is the one that destroys local ordering.
    check("yarn_keeps_high_freq", abs(yf[0] - pf[0]) < 1e-12,
          f"highest-frequency pair was scaled: yarn={yf[0]:.6g} plain={pf[0]:.6g}")
    # The lowest pair is only PARTIALLY interpolated here, and that is
    # reference-faithful rather than a bug: with original_max=65536 the
    # correction range runs to dim 33, past the end of the 32-element frequency
    # array, so the ramp never reaches 1. Assert the bracket, not equality —
    # an equality assertion here would be "fixed" by uniformly dividing every
    # frequency, which is exactly the mistake YaRN exists to avoid.
    check("yarn_interpolates_low_freq", pf[-1] / 16.0 < yf[-1] < pf[-1],
          f"lowest-frequency pair {yf[-1]:.6g} outside "
          f"({pf[-1] / 16.0:.6g}, {pf[-1]:.6g})")

    # ...and where the correction range DOES fit inside the array, the tail is
    # fully interpolated to exactly plain/factor. This pins the ramp's far end,
    # which the bracket above deliberately cannot.
    short = RopeSpec(head_dim=64, theta=10000.0, rope_type="yarn", scaling_factor=16.0,
                     original_max_position_embeddings=4096)
    sf = short.inv_freq()
    check("yarn_full_interpolation_when_range_fits",
          abs(sf[-1] - pf[-1] / 16.0) < 1e-15,
          f"tail not fully interpolated: {sf[-1]:.6g} vs {pf[-1] / 16.0:.6g}")
    check("yarn_monotone_ratio",
          bool(np.all(np.diff(yf / pf) <= 1e-12)),
          "the extrapolation->interpolation blend is not monotonic in dimension")

    # A yarn rotation must still round-trip, or strip/re-apply is broken for
    # every DeepSeek pair.
    ykeys = rng.standard_normal((2, 3, 6, 64)).astype(np.float32)
    ypos = np.array([0, 1, 100, 5000, 70000, 120000])
    yback = strip_rope(apply_rope(ykeys, yarn, ypos), yarn, ypos)
    check("yarn_roundtrip", float(np.abs(yback - ykeys).max()) < 1e-3,
          f"max abs err {float(np.abs(yback - ykeys).max()):.2e}")

    # ...and remain norm-preserving, which is what proves mscale is really 1.
    check("yarn_orthogonal",
          float(np.abs(np.linalg.norm(apply_rope(ykeys, yarn, ypos), axis=-1)
                       - np.linalg.norm(ykeys, axis=-1)).max()) < 1e-3,
          "yarn rotation is not norm-preserving — an mscale term is leaking in")

    # A scaled-cos/sin variant must be refused rather than silently mis-inverted.
    try:
        RopeSpec(head_dim=64, theta=1e4, rope_type="yarn", mscale=1.2)
        check("refuse_yarn_mscale", False, "accepted a non-unit yarn mscale")
    except ValueError as exc:
        check("refuse_yarn_mscale_reason", "mscale" in str(exc),
              f"refused for the wrong reason: {exc}")

    # 8. Fail closed on a schedule we still do not reproduce.
    try:
        RopeSpec(head_dim=8, theta=1e4, rope_type="longrope")
        check("refuse_unknown_rope", False, "accepted rope_type='longrope'")
    except ValueError as exc:
        check("refuse_unknown_rope_reason", "unsupported rope_type" in str(exc),
              f"refused, but not for the rope schedule: {exc}")

    # 8. Position count must match rows.
    try:
        apply_rope(keys, spec, [0, 1])
        check("refuse_bad_positions", False, "accepted 2 positions for 5 rows")
    except ValueError as exc:
        check("refuse_bad_positions_reason", "positions" in str(exc),
              f"refused, but not for the position count: {exc}")

    for f in failures:
        print(f"FAIL {f}")
    print(f"rope self-test: {'PASS' if not failures else 'FAIL'} "
          f"({8 - len(failures)}/8 groups ok)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
