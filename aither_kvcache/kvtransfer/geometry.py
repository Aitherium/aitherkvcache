"""KV geometry of a model, and what makes a (source, target) pair transferable.

## The two kinds of precondition, kept apart on purpose

**Definitional.** If it does not hold, the mapper is not "less accurate", it is
meaningless, and no amount of held-out R^2 can rescue it. There are exactly two:

* **Identical tokenization.** The map converts the KV of token position *i* in
  the source into the KV of token position *i* in the target. If the two models
  tokenize the same string differently, position *i* is not the same token, and
  the fit is regressing one document's representation onto a misaligned other.
  Nothing downstream can see this — the shapes agree, the fit converges, the
  R^2 is merely mediocre, and it reads as "this pair transfers poorly" rather
  than "these rows do not correspond". Every published pair shares a tokenizer
  by virtue of being one family, which is precisely why the requirement goes
  unstated.
* **A reproducible RoPE schedule.** See ``rope.py`` — an approximated frequency
  schedule fails silently.

**Empirical.** Everything else — same family, matching KV-head count, matching
head dimension, similar depth — is a prediction about how WELL a pair will
transfer, not whether the arithmetic is valid. Those are deliberately NOT
refusals here. They are recorded as regime flags on the pack and the decision
is handed to measured held-out fidelity, because that is the thing that
actually answers the question and it is cheap to measure.

That split matters for this fleet specifically: the locally available pair
(SmolLM2 135M -> 1.7B) has **mismatched KV-head counts**, 3 against 32, which
the published work lists as untested. A gate keyed on "heads must match" would
refuse the experiment that tells us whether heads need to match. A gate keyed
on measured R^2 runs it and reports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .rope import RopeSpec


def tokenizer_digest(tokenizer: object) -> str:
    """Stable digest of a tokenizer's token->id mapping and special tokens.

    Hashing the vocabulary rather than the tokenizer's name or files: two
    checkpoints in one family ship byte-different tokenizer.json files (added
    tokens, chat template, metadata) while agreeing perfectly on every id that
    matters. A file hash would refuse pairs that are in fact aligned.
    """
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if get_vocab is None:
        raise ValueError("tokenizer exposes no get_vocab(); cannot prove alignment")
    vocab: Dict[str, int] = get_vocab()
    h = hashlib.sha256()
    for token, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        h.update(f"{idx}\x00{token}\x00".encode("utf-8", errors="surrogatepass"))
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        h.update(f"{attr}={getattr(tokenizer, attr, None)}\x00".encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class KVGeometry:
    """The shape of one model's KV cache, plus the identity of its token stream."""

    model_id: str
    family: str                 # config.model_type — "llama", "qwen3", ...
    n_layers: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    rope_theta: float
    rope_type: str
    rope_scaling_factor: float
    tokenizer_sha256: str
    torch_dtype: str
    latent_attention: str = ""   # non-empty names the mechanism, e.g. "MLA"
    rope_key_dim: int = 0        # MLA decoupled RoPE key width (qk_rope_head_dim)

    @property
    def is_latent(self) -> bool:
        """True when this model caches a compressed latent instead of per-head K/V."""
        return bool(self.latent_attention)

    def roles(self) -> tuple[str, ...]:
        """The tensors this model's cache actually holds, per layer per token.

        A conventional model holds ``("k", "v")``. An MLA model holds
        ``("c", "kr")`` — a compressed latent and a decoupled RoPE key — and
        that is the whole difference. Treating roles as data rather than as the
        hardcoded pair ("k", "v") is what lets one mapper serve both: the ridge
        machinery is already role-agnostic, it was only the geometry that
        assumed two per-head tensors.

        Note ``kr`` is a SINGLE shared key across heads in MLA, not one per
        head — that is why it is its own role rather than a narrower "k".
        """
        return ("c", "kr") if self.is_latent else ("k", "v")

    def rotated_roles(self) -> tuple[str, ...]:
        """Which roles carry a position rotation and must be stripped/re-applied.

        For MLA this is the cleaner half of the design: the latent ``c`` is
        already position-free by construction (that is precisely why DeepSeek
        decouples the rope key), so only the small ``kr`` tensor rotates.
        Rotating ``c`` would corrupt it exactly as rotating V would.
        """
        return ("kr",) if self.is_latent else ("k",)

    def role_width(self, role: str) -> int:
        """Flattened width of one layer's ``role`` tensor for a single token."""
        if role in ("k", "v"):
            if self.is_latent:
                raise ValueError(
                    f"{self.model_id} is latent-attention; its roles are "
                    f"{self.roles()}, not per-head k/v"
                )
            return self.n_kv_heads * self.head_dim
        if role == "c":
            # num_key_value_heads is 1 for MLA and head_dim carries the latent
            # rank (512 on DeepSeek-V4-Flash), so the same product is correct.
            return self.n_kv_heads * self.head_dim
        if role == "kr":
            if not self.rope_key_dim:
                raise ValueError(
                    f"{self.model_id} declares no rope_key_dim; the decoupled key "
                    "width is required to lay out a latent cache"
                )
            return self.rope_key_dim
        raise ValueError(f"unknown cache role {role!r} for {self.model_id}")

    def total_width(self) -> int:
        """Flattened width of the WHOLE cache for one token, across all roles."""
        return self.n_layers * sum(self.role_width(r) for r in self.roles())

    @property
    def per_layer_width(self) -> int:
        """Flattened width of one layer's K (or V) for a single token."""
        return self.n_kv_heads * self.head_dim

    @property
    def flat_width(self) -> int:
        """Flattened width of ALL layers' K (or V) for a single token."""
        return self.n_layers * self.per_layer_width

    def rope_spec(self) -> RopeSpec:
        return RopeSpec(
            head_dim=self.head_dim,
            theta=self.rope_theta,
            rope_type=self.rope_type,
            scaling_factor=self.rope_scaling_factor,
        )

    @classmethod
    def from_hf(cls, model_id: str, config: object, tokenizer: object, dtype: str) -> "KVGeometry":
        head_dim = getattr(config, "head_dim", None)
        hidden = int(getattr(config, "hidden_size"))
        n_heads = int(getattr(config, "num_attention_heads"))
        if not head_dim:
            head_dim = hidden // n_heads
        n_kv = int(getattr(config, "num_key_value_heads", n_heads) or n_heads)
        scaling = getattr(config, "rope_scaling", None) or {}
        # Multi-head Latent Attention (DeepSeek V2/V3/V4) does not cache K and V
        # per head at all — it caches a compressed latent that each layer
        # decompresses. There is no (n_kv_heads, head_dim) tensor to map, so the
        # per-head mapper is not "less accurate" here, it is inapplicable. The
        # tell is a kv_lora_rank in the config; n_key_value_heads is still
        # populated, so a shape-only check sails straight past this.
        latent = ""
        if getattr(config, "kv_lora_rank", None):
            latent = f"MLA (kv_lora_rank={getattr(config, 'kv_lora_rank')})"
        elif str(getattr(config, "model_type", "")).startswith("deepseek_v"):
            latent = f"MLA (model_type={getattr(config, 'model_type')})"
        if latent:
            # A latent-attention model is refused on ARCHITECTURE by check_pair,
            # which is the more fundamental reason. Validating its rope schedule
            # first would raise and report the pair as merely "unresolved" —
            # true, but it buries the fact that no rope fix could ever make this
            # pair work. Record the geometry, let check_pair state the real one.
            spec = RopeSpec(head_dim=int(head_dim),
                            theta=float(getattr(config, "rope_theta", 10000.0)))
            scaling = {}
        else:
            spec = RopeSpec.from_hf_config(config)  # validates the schedule, may raise
        return cls(
            model_id=model_id,
            family=str(getattr(config, "model_type", "unknown")),
            n_layers=int(getattr(config, "num_hidden_layers")),
            n_kv_heads=n_kv,
            head_dim=int(head_dim),
            hidden_size=hidden,
            rope_theta=spec.theta,
            rope_type=str(scaling.get("rope_type") or scaling.get("type") or "default"),
            rope_scaling_factor=float(scaling.get("factor", 1.0) or 1.0),
            tokenizer_sha256=tokenizer_digest(tokenizer),
            torch_dtype=dtype,
            latent_attention=latent,
            rope_key_dim=int(getattr(config, 'qk_rope_head_dim', 0) or 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KVGeometry":
        fields = set(cls.__annotations__)
        missing = fields - set(data)
        if missing:
            raise ValueError(f"geometry record missing fields: {sorted(missing)}")
        return cls(**{k: data[k] for k in fields})


def source_roles_for(target_role: str, source: KVGeometry) -> tuple[str, ...]:
    """Which of the SOURCE's cache tensors feed one TARGET role.

    For conventional -> conventional this is the identity, and deliberately so:
    keys are predicted only from keys, values only from values. Letting them mix
    inflates training R^2 and buys nothing, because the two are different
    quantities with different statistics.

    **That rule inverts for a latent target, and the inversion is correct.**
    MLA's ``c`` is a JOINT compression of keys and values — the model
    reconstructs both from it via separate up-projections — so predicting ``c``
    from source keys alone discards exactly the value information the target
    will need. ``c`` therefore reads every source role.

    Getting this backwards is the expensive kind of wrong: a ``c`` fitted from
    keys only still trains, still loads, and produces a target that has
    forgotten what the values were, which surfaces as fluent but factually
    unmoored generation rather than as an error.
    """
    if target_role == "c":
        return tuple(source.roles())
    if target_role == "kr":
        # The decoupled rope key is positional/key-like; values cannot inform it.
        return ("kr",) if source.is_latent else ("k",)
    if target_role == "k":
        return ("c", "kr") if source.is_latent else ("k",)
    if target_role == "v":
        # Only the joint latent carries value information on a latent source.
        return ("c",) if source.is_latent else ("v",)
    raise ValueError(f"unknown target role {target_role!r}")


@dataclass
class PairVerdict:
    """Whether a pair can be fitted TODAY, and if not, precisely why not.

    Three states, and keeping them apart is the point of this class:

    * ``blocking`` — definitionally impossible. No implementation work changes it.
    * ``unimplemented`` — the pair is coherent, but needs a named capability this
      module does not have yet. This is a BACKLOG ITEM, not a wall.
    * ``regime_flags`` — fittable now, but in territory nobody has measured.

    The distinction is not pedantry; it was earned. MLA was originally reported
    here as "no tensor for a per-head map to touch", i.e. impossible. That is
    false: DeepSeek-V4-Flash caches 512 latent + 64 decoupled-RoPE dims per
    layer per token — a dense, well-defined tensor that is in fact **3x smaller**
    than a conventional cache of the same context. It needs a latent adapter,
    which is work. Reporting work as physics stops people doing it.
    """

    ok: bool
    blocking: List[str] = field(default_factory=list)
    unimplemented: List[str] = field(default_factory=list)
    regime_flags: List[str] = field(default_factory=list)

    def raise_if_blocked(self) -> None:
        if not self.ok:
            reasons = ["IMPOSSIBLE: " + b for b in self.blocking]
            reasons += ["UNIMPLEMENTED: " + u for u in self.unimplemented]
            raise ValueError(
                "source/target pair cannot be fitted yet: " + "; ".join(reasons)
            )


def check_pair(source: KVGeometry, target: KVGeometry) -> PairVerdict:
    """Definitional refusals, plus regime flags for anything merely unproven."""
    blocking: List[str] = []
    todo: List[str] = []
    flags: List[str] = []

    if source.tokenizer_sha256 != target.tokenizer_sha256:
        todo.append(
            f"cross-tokenizer realignment ({source.model_id} {source.tokenizer_sha256[:8]} "
            f"vs {target.model_id} {target.tokenizer_sha256[:8]}). The per-position map "
            "is INVALID as specified — row i is not the same token on both sides. It is "
            "not impossible: both tokenizers decode to the same string, so a "
            "character-span alignment defines which source tokens cover each target "
            "token, and the map becomes span-pooled rather than positional. That is a "
            "real research extension, not a config change."
        )
    if source.model_id == target.model_id:
        blocking.append("source and target are the same model — nothing to transfer")
    for side, geom in (("source", source), ("target", target)):
        if geom.latent_attention:
            todo.append(
                f"latent-KV adapter for {side} {geom.model_id} ({geom.latent_attention}). "
                "It caches a compressed latent plus a decoupled RoPE key rather than "
                "per-head K/V, so the per-HEAD map does not apply — but the latent is a "
                "dense per-token, per-layer tensor and the same ridge machinery maps to "
                "and from it. Strip/re-apply rotation on the decoupled key only; the "
                "latent is already position-free."
            )
    try:
        source.rope_spec()
        target.rope_spec()
    except ValueError as exc:
        todo.append(f"rope schedule support: {exc}")

    if source.family != target.family:
        flags.append(
            f"cross-family ({source.family} -> {target.family}): every published pair "
            "is same-family; expect poor fidelity and let the gate decide"
        )
    if source.head_dim != target.head_dim:
        flags.append(
            f"head_dim differs ({source.head_dim} -> {target.head_dim}): the affine map "
            "spans it, but no published pair tests it"
        )
    if source.n_kv_heads != target.n_kv_heads:
        flags.append(
            f"kv-head count differs ({source.n_kv_heads} -> {target.n_kv_heads}): "
            "explicitly listed as untested regime in the published work"
        )
    if abs(source.rope_theta - target.rope_theta) > 1e-6:
        flags.append(
            f"rope_theta differs ({source.rope_theta:g} -> {target.rope_theta:g}): "
            "strip/re-apply is therefore not a no-op"
        )
    return PairVerdict(ok=not (blocking or todo), blocking=blocking,
                       unimplemented=todo, regime_flags=flags)


# ── self-test ───────────────────────────────────────────────────────────────


def _geom(**over: Any) -> KVGeometry:
    base: Dict[str, Any] = dict(
        model_id="m/a", family="llama", n_layers=24, n_kv_heads=8, head_dim=64,
        hidden_size=2048, rope_theta=10000.0, rope_type="default",
        rope_scaling_factor=1.0, tokenizer_sha256="a" * 64, torch_dtype="float32",
    )
    base.update(over)
    return KVGeometry(**base)


def _self_test() -> int:
    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(f"{name}: {detail}")

    src = _geom(model_id="small", n_layers=30, n_kv_heads=3)
    tgt = _geom(model_id="big", n_layers=24, n_kv_heads=32)

    v = check_pair(src, tgt)
    check("mismatched_heads_allowed", v.ok,
          f"refused a pair it should merely flag: {v.blocking}")
    check("mismatched_heads_flagged",
          any("kv-head count" in f for f in v.regime_flags),
          f"no regime flag raised: {v.regime_flags}")

    # Definitional refusal: different tokenizers.
    v2 = check_pair(src, _geom(model_id="big", tokenizer_sha256="b" * 64))
    check("tokenizer_mismatch_blocks", not v2.ok, "accepted a tokenizer mismatch")
    check("tokenizer_reason_named", any("tokenizer" in u for u in v2.unimplemented),
          f"tokenizer reason not in unimplemented: {v2.unimplemented}")
    check("tokenizer_is_not_impossible", not v2.blocking,
          "a tokenizer mismatch was filed as definitionally impossible; it needs "
          "realignment work, and calling work physics stops people doing it")

    # MLA is unimplemented, never impossible.
    mla = _geom(model_id="mla", latent_attention="MLA (kv_lora_rank=512)")
    v_mla = check_pair(src, mla)
    check("mla_not_fittable_today", not v_mla.ok, "accepted an MLA target today")
    check("mla_is_unimplemented", any("latent-KV adapter" in u for u in v_mla.unimplemented),
          f"MLA not filed as unimplemented: {v_mla.unimplemented}")
    check("mla_is_not_impossible", not v_mla.blocking,
          "MLA was filed as definitionally impossible — it is a layout, not a wall")

    # MUTATION GUARD: the tokenizer rule must be the thing doing the blocking,
    # not an unrelated field difference. Same pair, matching tokenizers, passes.
    v3 = check_pair(src, _geom(model_id="big"))
    check("mutation_guard_tokenizer", v3.ok,
          f"an otherwise-identical pair was still refused: {v3.blocking}")

    v4 = check_pair(src, src)
    check("self_pair_blocks", not v4.ok, "accepted source == target")

    # Widths are what the capture layout depends on; a wrong one silently
    # misaligns every column of the Gram.
    check("flat_width", src.flat_width == 30 * 3 * 64, f"got {src.flat_width}")
    check("per_layer_width", tgt.per_layer_width == 32 * 64, f"got {tgt.per_layer_width}")

    # Cache roles are data, not a hardcoded ("k", "v") pair — this is what lets
    # one mapper serve both conventional and latent models.
    ds = _geom(model_id="ds", n_layers=43, n_kv_heads=1, head_dim=512,
               latent_attention="MLA (model_type=deepseek_v4)", rope_key_dim=64)
    check("conventional_roles", src.roles() == ("k", "v"), f"got {src.roles()}")
    check("latent_roles", ds.roles() == ("c", "kr"), f"got {ds.roles()}")
    check("latent_widths",
          (ds.role_width("c"), ds.role_width("kr")) == (512, 64),
          f"got {(ds.role_width('c'), ds.role_width('kr'))}")
    check("latent_total_width", ds.total_width() == 43 * (512 + 64),
          f"got {ds.total_width()}")

    # Only the decoupled key rotates. Rotating the latent would corrupt it
    # exactly as rotating V would, and it is the kind of error that produces a
    # plausible-looking fit rather than a crash.
    check("latent_rotated_roles", ds.rotated_roles() == ("kr",),
          f"got {ds.rotated_roles()}")
    check("conventional_rotated_roles", src.rotated_roles() == ("k",),
          f"got {src.rotated_roles()}")

    # Asking a latent model for per-head k/v must raise rather than silently
    # returning n_kv_heads*head_dim, which happens to be a real number (512)
    # and would lay out the cache wrongly with no error anywhere.
    try:
        ds.role_width("k")
        check("latent_refuses_kv_role", False, "returned a width for role 'k'")
    except ValueError as exc:
        check("latent_refuses_kv_reason", "latent-attention" in str(exc),
              f"refused for the wrong reason: {exc}")
    try:
        _geom(model_id="ds2", latent_attention="MLA", rope_key_dim=0).role_width("kr")
        check("missing_rope_key_dim_refused", False, "accepted rope_key_dim=0")
    except ValueError as exc:
        check("missing_rope_key_dim_reason", "rope_key_dim" in str(exc),
              f"refused for the wrong reason: {exc}")

    # Role pairing. Conventional keeps K and V separate; a latent target reads
    # BOTH, because MLA's c is a joint compression of keys and values.
    check("conventional_pairing",
          source_roles_for("k", src) == ("k",) and source_roles_for("v", src) == ("v",),
          "conventional pairing mixed K and V")
    check("latent_target_reads_both", source_roles_for("c", src) == ("k", "v"),
          f"latent c read only {source_roles_for('c', src)} — it would forget values")
    check("rope_key_from_keys", source_roles_for("kr", src) == ("k",),
          f"got {source_roles_for('kr', src)}")
    check("latent_source_to_v", source_roles_for("v", ds) == ("c",),
          f"got {source_roles_for('v', ds)}")
    check("latent_source_to_k", source_roles_for("k", ds) == ("c", "kr"),
          f"got {source_roles_for('k', ds)}")
    try:
        source_roles_for("q", src)
        check("unknown_role_refused", False, "accepted target role 'q'")
    except ValueError:
        check("unknown_role_refused", True)

    round_tripped = KVGeometry.from_dict(json.loads(json.dumps(src.to_dict())))
    check("roundtrip", round_tripped == src, "geometry did not survive json round trip")

    try:
        KVGeometry.from_dict({"model_id": "x"})
        check("partial_dict_refused", False, "accepted a geometry missing most fields")
    except ValueError as exc:
        check("partial_dict_reason", "missing fields" in str(exc),
              f"refused, but not for missing fields: {exc}")

    for f in failures:
        print(f"FAIL {f}")
    print(f"geometry self-test: {'PASS' if not failures else 'FAIL'} ({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
