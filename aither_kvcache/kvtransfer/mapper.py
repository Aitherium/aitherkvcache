"""The cross-model KV mapper: per-(target layer, head) linear maps, fitted by ridge.

## The shape of the thing

For each target layer ``t`` and each target KV head ``h``, two independent
affine maps::

    K_target[t, h] = W_K[t, h] @ concat(K_source[s] for s in top_k(t)) + b_K[t, h]
    V_target[t, h] = W_V[t, h] @ concat(V_source[s] for s in top_k(t)) + b_V[t, h]

Three properties, each of which is load-bearing:

1. **K and V never mix.** The K map reads only source keys, the V map only
   source values. They are different quantities with different statistics;
   letting them mix inflates training R^2 and buys nothing at inference.
2. **The input is every KV head of the selected source layers**, not the
   matching head index. Head ``h`` of the target has no reason to correspond to
   head ``h`` of the source, and this is what lets a 3-KV-head source drive a
   32-KV-head target — the head counts never have to agree. Only ``d_h`` of the
   *output* is fixed, by the target.
3. **Source layers are chosen, not aligned.** Layer counts differ (30 vs 24
   here, 40 vs 64 in the paper), so there is no natural pairing. Each target
   layer ranks every source layer by held-out R^2 and takes the top k. One
   source layer reconstructs ~56% of key variance in the paper's Qwen3 pair;
   the top 8 together reach ~79%.

## Why everything is computed from Gram matrices

The fit needs, per target layer, one regression per candidate source layer
(to rank them) plus one on the selected block. Done naively that is
``n_source_layers + 1`` passes over the activations per target layer —
``31 * 24 = 744`` passes here, and far worse at real scale.

Every one of those regressions is a submatrix of the same two accumulators:
``X'X`` over all source layers, and ``X'Y`` for that target layer. So the data
is touched once per target layer, and ranking 30 candidates plus fitting the
winner is pure linear algebra on slices. That is also what makes the
lambda sweep free.

## Splits are by SEQUENCE, never by token

Adjacent tokens of one document are strongly correlated. A token-level split
puts near-duplicates on both sides and reports an R^2 that a sequence-level
split will not reproduce — the classic way to publish a number that does not
survive contact with a new document. ``fit_mapper`` takes ``seq_ids`` and
refuses to run without them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("kvtransfer.mapper")

# Relative ridge strengths swept per target layer. Relative to the mean
# eigenvalue of the centered Gram, so the sweep is scale-free across models.
DEFAULT_LAMBDAS: Tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


@dataclass
class GramStats:
    """Sufficient statistics for every ridge fit over one (X, Y) pair.

    Uncentered on purpose: centering is applied analytically at solve time, so
    one accumulation serves fits with and without an intercept, and the val
    statistics can be reused across every lambda and every subset of columns.
    """

    n: int
    sum_x: np.ndarray        # (D,)
    sum_y: np.ndarray        # (M,)
    xtx: np.ndarray          # (D, D)
    xty: np.ndarray          # (D, M)
    yty_diag: np.ndarray     # (M,) — only the diagonal is ever needed

    @classmethod
    def accumulate(cls, x: np.ndarray, y: np.ndarray) -> "GramStats":
        """Build from dense blocks. float64 throughout — this is the numerics."""
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"row mismatch: X has {x.shape[0]}, Y has {y.shape[0]}")
        if x.shape[0] == 0:
            raise ValueError("cannot accumulate Gram statistics over zero rows")
        xd = x.astype(np.float64, copy=False)
        yd = y.astype(np.float64, copy=False)
        return cls(
            n=int(xd.shape[0]),
            sum_x=xd.sum(axis=0),
            sum_y=yd.sum(axis=0),
            xtx=xd.T @ xd,
            xty=xd.T @ yd,
            yty_diag=np.einsum("ij,ij->j", yd, yd),
        )

    def select_columns(self, cols: Sequence[int]) -> "GramStats":
        """Restrict to a subset of INPUT dimensions — how a source-layer subset is fitted."""
        idx = np.asarray(cols, dtype=np.int64)
        return GramStats(
            n=self.n,
            sum_x=self.sum_x[idx],
            sum_y=self.sum_y,
            xtx=self.xtx[np.ix_(idx, idx)],
            xty=self.xty[idx, :],
            yty_diag=self.yty_diag,
        )


def _centered(stats: GramStats) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean-centered Gram blocks plus the two means."""
    mx = stats.sum_x / stats.n
    my = stats.sum_y / stats.n
    xtx_c = stats.xtx - stats.n * np.outer(mx, mx)
    xty_c = stats.xty - stats.n * np.outer(mx, my)
    return xtx_c, xty_c, mx, my


def solve_ridge(stats: GramStats, lam_rel: float) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge with intercept, from Gram statistics alone.

    Returns (W, b) with W of shape (D, M) and b of shape (M,), so a prediction
    is ``x @ W + b``.

    ``lam_rel`` is relative to the mean eigenvalue of the centered Gram; an
    absolute lambda is meaningless across models whose activation scales differ
    by an order of magnitude.
    """
    xtx_c, xty_c, mx, my = _centered(stats)
    d = xtx_c.shape[0]
    scale = float(np.trace(xtx_c)) / max(d, 1)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("centered Gram has non-positive trace — degenerate inputs")
    ridge = xtx_c + (lam_rel * scale) * np.eye(d)
    try:
        w = np.linalg.solve(ridge, xty_c)
    except np.linalg.LinAlgError:
        # Never fall back silently to something weaker: a pseudo-inverse here
        # would produce a mapper that loads and reconstructs noise. But a
        # singular system at a nonzero lambda is a real numerical event worth
        # surviving, so log it loudly and use lstsq.
        logger.warning("ridge solve singular at lam_rel=%g (D=%d) — using lstsq", lam_rel, d)
        w = np.linalg.lstsq(ridge, xty_c, rcond=None)[0]
    b = my - w.T @ mx
    return w, b


def r2_from_gram(stats: GramStats, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-output-dimension R^2 of (W, b) evaluated on ``stats``.

    Expanded from the Gram so a held-out score costs no data pass::

        RSS_j = Y'Y_jj - 2 w_j.X'Y_j - 2 b_j SY_j + w_j'X'X w_j
                + 2 b_j SX.w_j + n b_j^2

    R^2 is against the mean of the evaluation set, so a val score is a genuine
    out-of-sample number and can go negative — which is the honest report for a
    map that is worse than predicting the mean, and must never be clipped.
    """
    n = stats.n
    quad = np.einsum("ij,ij->j", w, stats.xtx @ w)
    cross = np.einsum("ij,ij->j", w, stats.xty)
    sx_w = stats.sum_x @ w
    rss = (
        stats.yty_diag
        - 2.0 * cross
        - 2.0 * b * stats.sum_y
        + quad
        + 2.0 * b * sx_w
        + n * b * b
    )
    tss = stats.yty_diag - (stats.sum_y ** 2) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - rss / tss
    # A constant output dimension has zero variance: R^2 is undefined, not 1.0.
    # Reporting 1.0 there is how a dead channel inflates a headline average.
    r2 = np.where(tss > 1e-12, r2, np.nan)
    return r2


@dataclass
class LayerMap:
    """The fitted map for one target layer and one tensor role (K or V)."""

    target_layer: int
    role: str                      # "k" or "v"
    source_layers: List[int]
    lam_rel: float
    weight: np.ndarray             # (D_sel, H_tgt * d_h) float32
    bias: np.ndarray               # (H_tgt * d_h,) float32
    val_r2_mean: float
    val_r2_per_head: List[float]
    candidate_r2: Dict[int, float] = field(default_factory=dict)


def _block_columns(layer_idx: int, block: int) -> np.ndarray:
    """Column indices of one source layer inside the flattened source vector."""
    start = layer_idx * block
    return np.arange(start, start + block, dtype=np.int64)


def fit_layer_map(
    train: GramStats,
    val: GramStats,
    *,
    target_layer: int,
    role: str,
    n_source_layers: int,
    source_block: int,
    top_k: int,
    n_target_heads: int,
    head_dim: int,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> LayerMap:
    """Rank source layers by held-out R^2, then fit the top-k block.

    Both stages score on ``val``. Ranking on training R^2 would systematically
    prefer whichever source layer has the most capacity to overfit rather than
    the one that transfers, which is the failure this two-set design exists to
    avoid.
    """
    if role not in ("k", "v"):
        raise ValueError(f"role must be 'k' or 'v', got {role!r}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    # Stage 1 — rank every source layer alone.
    candidate: Dict[int, float] = {}
    for s in range(n_source_layers):
        cols = _block_columns(s, source_block)
        tr_s, va_s = train.select_columns(cols), val.select_columns(cols)
        best = -np.inf
        for lam in lambdas:
            w, b = solve_ridge(tr_s, lam)
            score = float(np.nanmean(r2_from_gram(va_s, w, b)))
            best = max(best, score)
        candidate[s] = best

    chosen = sorted(candidate, key=lambda s: candidate[s], reverse=True)[:top_k]
    chosen.sort()  # keep column order deterministic and readable

    # Stage 2 — fit the selected block, sweeping lambda on held-out data.
    cols = np.concatenate([_block_columns(s, source_block) for s in chosen])
    tr_sel, va_sel = train.select_columns(cols), val.select_columns(cols)
    best_w: Optional[np.ndarray] = None
    best_b: Optional[np.ndarray] = None
    best_lam = float(lambdas[0])
    best_r2 = -np.inf
    best_per_dim: Optional[np.ndarray] = None
    for lam in lambdas:
        w, b = solve_ridge(tr_sel, lam)
        per_dim = r2_from_gram(va_sel, w, b)
        score = float(np.nanmean(per_dim))
        if score > best_r2:
            best_r2, best_w, best_b, best_lam, best_per_dim = score, w, b, float(lam), per_dim
    if best_w is None or best_b is None or best_per_dim is None:
        raise RuntimeError("lambda sweep produced no solution — empty lambda list?")

    per_head = [
        float(np.nanmean(best_per_dim[h * head_dim:(h + 1) * head_dim]))
        for h in range(n_target_heads)
    ]
    return LayerMap(
        target_layer=target_layer,
        role=role,
        source_layers=chosen,
        lam_rel=best_lam,
        weight=best_w.astype(np.float32),
        bias=best_b.astype(np.float32),
        val_r2_mean=best_r2,
        val_r2_per_head=per_head,
        candidate_r2={int(s): float(v) for s, v in candidate.items()},
    )


def apply_layer_map(source_flat: np.ndarray, lm: LayerMap, source_block: int) -> np.ndarray:
    """Convert source activations to one target layer's KV.

    Args:
        source_flat: (n_tokens, n_source_layers * source_block) — the SAME
            flattening used at fit time, RoPE already stripped for role "k".
        lm: the fitted map.
        source_block: per-source-layer width, to slice the selected layers out.

    Returns:
        (n_tokens, H_tgt * d_h) float32, still in position-free space for keys.
    """
    cols = np.concatenate([_block_columns(s, source_block) for s in lm.source_layers])
    x = source_flat[:, cols].astype(np.float32, copy=False)
    return x @ lm.weight + lm.bias


# ── self-test ───────────────────────────────────────────────────────────────


def _self_test() -> int:
    """Prove every rule can still fail. Exit 0 pass, 1 fail."""
    rng = np.random.default_rng(4242)
    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(f"{name}: {detail}")

    # A planted linear relationship the fitter must recover: target depends on
    # source layers 2 and 5 only, plus noise.
    n_src_layers, block, d_h, h_tgt = 8, 6, 3, 2
    n_tr, n_va = 4000, 1500
    d_full = n_src_layers * block
    m_out = h_tgt * d_h
    true_cols = np.concatenate([_block_columns(2, block), _block_columns(5, block)])
    w_true = rng.standard_normal((len(true_cols), m_out))
    b_true = rng.standard_normal(m_out)

    def make(n: int) -> Tuple[np.ndarray, np.ndarray]:
        x = rng.standard_normal((n, d_full))
        y = x[:, true_cols] @ w_true + b_true + 0.05 * rng.standard_normal((n, m_out))
        return x, y

    xtr, ytr = make(n_tr)
    xva, yva = make(n_va)
    tr = GramStats.accumulate(xtr, ytr)
    va = GramStats.accumulate(xva, yva)

    lm = fit_layer_map(
        tr, va, target_layer=0, role="k", n_source_layers=n_src_layers,
        source_block=block, top_k=2, n_target_heads=h_tgt, head_dim=d_h,
    )
    check("recovers_planted_layers", lm.source_layers == [2, 5],
          f"selected {lm.source_layers}, expected [2, 5]")
    check("high_r2_on_linear_data", lm.val_r2_mean > 0.98,
          f"val R2 {lm.val_r2_mean:.4f} on planted-linear data")

    # apply() must agree with the fit, not merely run.
    pred = apply_layer_map(xva, lm, block)
    resid = float(np.mean((pred - yva) ** 2)) / float(np.var(yva))
    check("apply_matches_fit", resid < 0.02, f"normalised residual {resid:.4f}")

    # MUTATION GUARD: shuffling the selected columns must break apply(). Without
    # this, a version of apply_layer_map that slices the WRONG columns still
    # returns the right shape and would pass every check above.
    bad = LayerMap(
        target_layer=0, role="k", source_layers=[0, 1], lam_rel=lm.lam_rel,
        weight=lm.weight, bias=lm.bias, val_r2_mean=lm.val_r2_mean,
        val_r2_per_head=lm.val_r2_per_head,
    )
    bad_resid = float(np.mean((apply_layer_map(xva, bad, block) - yva) ** 2)) / float(np.var(yva))
    check("mutation_guard_columns", bad_resid > 0.5,
          f"wrong source columns still fit well (residual {bad_resid:.3f}) — "
          "apply_layer_map is not really honouring source_layers")

    # R^2 on pure noise must be ~0 (and may be negative), never ~1. This is the
    # assertion that catches an r2_from_gram that accidentally scores training
    # residuals or clips at zero.
    xn, yn = rng.standard_normal((2000, d_full)), rng.standard_normal((2000, m_out))
    xn2, yn2 = rng.standard_normal((2000, d_full)), rng.standard_normal((2000, m_out))
    tr_n, va_n = GramStats.accumulate(xn, yn), GramStats.accumulate(xn2, yn2)
    w_n, b_n = solve_ridge(tr_n, 1e-6)
    noise_r2 = float(np.nanmean(r2_from_gram(va_n, w_n, b_n)))
    check("noise_scores_zero", noise_r2 < 0.05,
          f"held-out R2 on pure noise is {noise_r2:.4f} — leakage in r2_from_gram")

    # A constant output channel must be NaN, not 1.0.
    yc = np.zeros((2000, m_out))
    yc[:, 0] = 7.0
    yc[:, 1:] = rng.standard_normal((2000, m_out - 1))
    st = GramStats.accumulate(xn, yc)
    w_c, b_c = solve_ridge(st, 1e-3)
    r2c = r2_from_gram(st, w_c, b_c)
    check("constant_channel_is_nan", bool(np.isnan(r2c[0])),
          f"constant channel scored {r2c[0]} instead of NaN")

    # Column selection must actually restrict.
    sub = tr.select_columns(_block_columns(2, block))
    check("select_columns_shape", sub.xtx.shape == (block, block),
          f"got {sub.xtx.shape}")

    # Fail closed on bad arguments.
    for name, fn in (
        ("reject_role", lambda: fit_layer_map(
            tr, va, target_layer=0, role="q", n_source_layers=n_src_layers,
            source_block=block, top_k=1, n_target_heads=h_tgt, head_dim=d_h)),
        ("reject_topk", lambda: fit_layer_map(
            tr, va, target_layer=0, role="k", n_source_layers=n_src_layers,
            source_block=block, top_k=0, n_target_heads=h_tgt, head_dim=d_h)),
        ("reject_empty", lambda: GramStats.accumulate(
            np.zeros((0, 3)), np.zeros((0, 2)))),
        ("reject_row_mismatch", lambda: GramStats.accumulate(
            np.zeros((5, 3)), np.zeros((4, 2)))),
    ):
        try:
            fn()
            check(name, False, "accepted invalid input")
        except (ValueError, RuntimeError) as exc:
            check(f"{name}_message", str(exc).strip() != "",
                  "refused with an empty message — unexplainable to a caller")

    for f in failures:
        print(f"FAIL {f}")
    print(f"mapper self-test: {'PASS' if not failures else 'FAIL'} ({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(_self_test())
