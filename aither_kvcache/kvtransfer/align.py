"""Character-span alignment: map a source token stream onto a target's.

The positional mapper sends source row *i* to target row *i*, which is only
meaningful when both models tokenize identically. Two models with DIFFERENT
tokenizers still read the same string, so the string is the shared coordinate:
every token of either model covers a span of characters, and a target token's
row can be predicted from the source tokens that cover the same characters.

This module builds that correspondence as a CSR pooling matrix ``P`` of shape
``(n_target_tokens, n_source_tokens)``; aligned source rows are ``P @ X``. Each
row averages the source tokens assigned to one target token.

## The causal rule is the whole safety argument

A source token may be pooled into target token *j* only if its character END is
``<= char_end(j)``. A source token that ends after target *j* carries text the
target has not read yet at position *j*; pooling it would put FUTURE text into
the past of the cache. That is not an accuracy problem, it is a leak, and it
inflates every arm of the acceptance measurement at once (translated AND
control), so it cannot be seen from the headline numbers. ``causal_violations``
re-derives the rule from offsets alone, independently of the builder, and the
evaluator refuses to score a cache whose matrix violates it.

## Byte-level BPE makes character offsets ambiguous

A byte-level tokenizer splits one multi-byte character across tokens and
reports the SAME character span for each piece (``🚀`` -> three tokens, all
``(23, 24)``), and a piece can straddle (`` 🚀`` -> ``(22, 24)``). Overlap on
raw character spans would then pool both pieces into each other and the
same-tokenizer matrix would not be the identity. So every run of mutually
overlapping tokens is replaced by an ordered partition of its character range
into equal fractional pieces (``fractional_spans``). Identical offset lists
produce identical partitions, which is what makes the same-tokenizer matrix
EXACTLY the identity. Fractional coordinates only decide locality; causality is
always checked on the integer character end, and the fractional end is required
to be ``<=`` as well, which can only make the rule stricter.

## Fallback

A target token with no eligible source token (e.g. the first byte-piece of a
character the source tokenizes whole) carries the previous target row's pooling,
which is causal because target ends are non-decreasing. A leading target token
with nothing before it gets an all-zero row. Both are counted, never silent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Closed enum. A pack whose alignment method is not one of these is refused.
ALIGN_METHODS = ("positional", "span")

#: Bumped whenever the pooling rule changes; part of the digest so a pack fitted
#: under one rule never loads under another.
SPAN_RULE_VERSION = 1

#: Width given to a zero-width run so it still occupies a position in the
#: fractional coordinate. Small next to any real character (width 1).
_ZERO_WIDTH = 1e-3

Offsets = Sequence[Tuple[int, int]]


@dataclass(frozen=True)
class AlignmentSpec:
    """How source rows were made to correspond to target rows.

    ``digest`` is stored in the pack manifest. Anything that changes which
    source rows feed which target row is a field here, so it changes the digest.
    """

    method: str
    source_tokenizer_sha256: str
    target_tokenizer_sha256: str
    causal: bool = True
    pooling: str = "mean"
    fallback: str = "carry_previous"
    add_special_tokens: bool = False
    rule_version: int = SPAN_RULE_VERSION

    def __post_init__(self) -> None:
        if self.method not in ALIGN_METHODS:
            raise ValueError(f"alignment method {self.method!r} not in {ALIGN_METHODS}")

    @classmethod
    def span(cls, source_tokenizer_sha256: str, target_tokenizer_sha256: str
             ) -> "AlignmentSpec":
        """The spec this build writes for a span-aligned pack."""
        return cls(method="span", source_tokenizer_sha256=source_tokenizer_sha256,
                   target_tokenizer_sha256=target_tokenizer_sha256)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlignmentSpec":
        known = set(cls.__annotations__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"alignment spec has unknown fields {sorted(unknown)}")
        return cls(**data)

    def digest(self) -> str:
        """sha256 over the canonical JSON of every field."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fractional_spans(offsets: Offsets) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frac_start, frac_end, char_end) per token; fractional spans are disjoint.

    Consecutive tokens whose character spans overlap (byte pieces of one
    character, or a piece straddling into it) form a run; the run's character
    range is cut into equal ordered pieces, one per token. Disjoint, ordered,
    and a pure function of the offsets — identical offsets give identical spans.
    """
    n = len(offsets)
    starts = np.asarray([int(a) for a, _ in offsets], dtype=np.int64)
    ends = np.asarray([int(b) for _, b in offsets], dtype=np.int64)
    if n and (np.any(ends < starts) or np.any(np.diff(ends) < 0)):
        raise ValueError("offsets are not ordered spans (end < start, or ends decrease)")
    f0 = np.zeros(n, dtype=np.float64)
    f1 = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        lo, hi = int(starts[i]), int(ends[i])
        j = i + 1
        while j < n:
            a, b = int(starts[j]), int(ends[j])
            zero = a == b or lo == hi
            if a < hi or (zero and a <= hi):
                hi = max(hi, b)
                j += 1
            else:
                break
        width = float(hi - lo) if hi > lo else _ZERO_WIDTH
        count = j - i
        for r in range(count):
            f0[i + r] = lo + width * r / count
            f1[i + r] = lo + width * (r + 1) / count
        i = j
    return f0, f1, ends


@dataclass
class SpanCSR:
    """A pooling matrix in CSR form, plus the counts that make it auditable."""

    indptr: np.ndarray     # (n_tgt + 1,) int64
    indices: np.ndarray    # (nnz,) int64 source token index
    data: np.ndarray       # (nnz,) float32 weight
    n_src: int
    n_tgt: int
    n_carried: int = 0     # target rows filled by the carry-previous fallback
    n_empty: int = 0       # leading target rows with nothing to carry (zero rows)
    n_unused_src: int = 0  # source tokens pooled into no target row

    def to_dense(self) -> np.ndarray:
        dense = np.zeros((self.n_tgt, self.n_src), dtype=np.float32)
        rows = np.repeat(np.arange(self.n_tgt), np.diff(self.indptr))
        dense[rows, self.indices] = self.data
        return dense

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Pool ``x`` (n_src, D) onto target rows -> (n_tgt, D) in x's dtype.

        Dense matmul in float32: a window is a few hundred tokens, so the
        matrix is tiny, and an identity matrix reproduces ``x`` bit-exactly.
        """
        if x.shape[0] != self.n_src:
            raise ValueError(f"pooling {x.shape[0]} source rows with a {self.n_src}-row CSR")
        out = self.to_dense() @ x.astype(np.float32, copy=False)
        return out.astype(x.dtype, copy=False)

    def is_identity(self) -> bool:
        return (self.n_src == self.n_tgt
                and np.array_equal(self.indptr, np.arange(self.n_tgt + 1))
                and np.array_equal(self.indices, np.arange(self.n_tgt))
                and bool(np.all(self.data == 1.0)))

    def stats(self) -> Dict[str, int]:
        return {"n_src": self.n_src, "n_tgt": self.n_tgt, "nnz": int(self.indices.size),
                "n_carried": self.n_carried, "n_empty": self.n_empty,
                "n_unused_src": self.n_unused_src}


def build_span_csr(src_offsets: Offsets, tgt_offsets: Offsets, *,
                   causal: bool = True) -> SpanCSR:
    """Causal span pooling from source tokens onto target tokens.

    ``causal=False`` exists ONLY so the leak canary can prove the evaluation
    detects a leak. Nothing that writes or serves a pack may pass it.
    """
    sf0, sf1, s_end = fractional_spans(src_offsets)
    tf0, tf1, t_end = fractional_spans(tgt_offsets)
    n_src, n_tgt = len(sf0), len(tf0)
    indptr = [0]
    indices: List[int] = []
    data: List[float] = []
    used = np.zeros(n_src, dtype=bool)
    carried = empty = 0
    prev: Optional[Tuple[List[int], List[float]]] = None
    for j in range(n_tgt):
        # Candidates overlap target j in fractional coordinates; both arrays are
        # sorted, so they form one contiguous index range.
        lo = int(np.searchsorted(sf1, tf0[j], side="right"))
        hi = int(np.searchsorted(sf0, tf1[j], side="left"))
        row = [s for s in range(lo, hi) if sf0[s] < tf1[j] and sf1[s] > tf0[j]]
        if causal:
            row = [s for s in row if s_end[s] <= t_end[j] and sf1[s] <= tf1[j] + 1e-9]
        if row:
            w = 1.0 / len(row)
            cur = (row, [w] * len(row))
            used[row] = True
        elif prev is not None:
            cur = prev
            carried += 1
        else:
            cur = ([], [])
            empty += 1
        indices.extend(cur[0])
        data.extend(cur[1])
        indptr.append(len(indices))
        prev = cur if cur[0] else prev
    return SpanCSR(
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        data=np.asarray(data, dtype=np.float32),
        n_src=n_src, n_tgt=n_tgt, n_carried=carried, n_empty=empty,
        n_unused_src=int((~used).sum()),
    )


def causal_violations(csr: SpanCSR, src_offsets: Offsets, tgt_offsets: Offsets
                      ) -> List[Tuple[int, int]]:
    """(target j, source s) pairs where s ends AFTER target j — i.e. a leak.

    Deliberately independent of ``build_span_csr``: it reads only the matrix and
    the raw integer offsets, so a bug in the builder cannot also hide here.
    """
    s_end = np.asarray([int(b) for _, b in src_offsets], dtype=np.int64)
    t_end = np.asarray([int(b) for _, b in tgt_offsets], dtype=np.int64)
    bad: List[Tuple[int, int]] = []
    for j in range(csr.n_tgt):
        for k in range(int(csr.indptr[j]), int(csr.indptr[j + 1])):
            s = int(csr.indices[k])
            if csr.data[k] != 0 and s_end[s] > t_end[j]:
                bad.append((j, s))
    return bad


def require_fast_tokenizer(tokenizer: object, side: str) -> None:
    """Span alignment needs offsets, and only fast tokenizers return them."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            f"span alignment needs a FAST tokenizer for the {side} "
            f"({type(tokenizer).__name__} is not): character offsets come from "
            "return_offsets_mapping, which slow tokenizers do not implement"
        )


def encode_with_offsets(tokenizer: object, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """(ids, offsets) for ``text`` without special tokens."""
    enc = tokenizer(text, add_special_tokens=False,  # type: ignore[operator]
                    return_offsets_mapping=True)
    ids = [int(i) for i in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    if len(ids) != len(offsets):
        raise RuntimeError("tokenizer returned a different number of ids and offsets")
    return ids, offsets


Spans = Tuple[np.ndarray, np.ndarray, np.ndarray]


def source_window(src: "Offsets | Spans", tgt: "Offsets | Spans", t_lo: int, t_hi: int
                  ) -> Tuple[int, int]:
    """Source index range [s_lo, s_hi) that feeds target tokens [t_lo, t_hi).

    Cut by characters: starts at the source token that covers the target
    window's first character (it may begin a little EARLIER — past text, never
    a leak) and ends at the last source token that is complete by the window's
    final character, in both integer and fractional terms. ``src``/``tgt`` are
    raw offsets or a precomputed ``fractional_spans`` triple (a long document is
    cut into many windows; recomputing per window is quadratic).
    """
    sf0, sf1, s_end = _as_spans(src)
    tf0, tf1, t_end = _as_spans(tgt)
    start_f = tf0[t_lo]
    end_c, end_f = t_end[t_hi - 1], tf1[t_hi - 1]
    s_lo = int(np.searchsorted(sf1, start_f, side="right"))
    s_hi = s_lo
    while s_hi < len(sf0) and s_end[s_hi] <= end_c and sf1[s_hi] <= end_f + 1e-9:
        s_hi += 1
    return s_lo, s_hi


def _as_spans(obj: "Offsets | Spans") -> Spans:
    if (isinstance(obj, tuple) and len(obj) == 3
            and all(isinstance(a, np.ndarray) for a in obj)):
        return obj  # type: ignore[return-value]
    return fractional_spans(obj)  # type: ignore[arg-type]


# ── self-test ───────────────────────────────────────────────────────────────


def _random_offsets(rng: np.random.Generator, n_chars: int, max_tok: int,
                    dup_p: float) -> List[Tuple[int, int]]:
    """Synthetic byte-BPE-like offsets: contiguous spans plus duplicated pieces."""
    out: List[Tuple[int, int]] = []
    pos = 0
    while pos < n_chars:
        width = int(rng.integers(1, max_tok + 1))
        span = (pos, min(n_chars, pos + width))
        out.append(span)
        if rng.random() < dup_p:   # a character split into byte pieces
            last = (span[1] - 1, span[1])
            out.extend([last] * int(rng.integers(1, 3)))
        pos = span[1]
    return out


def _self_test() -> int:
    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(f"{name}: {detail}")

    rng = np.random.default_rng(1234)

    # Same tokenization -> exactly the identity, byte pieces included.
    byte_bpe = [(0, 5), (5, 6), (6, 12), (22, 24), (23, 24), (23, 24), (24, 25),
                (24, 25), (25, 28), (28, 28), (28, 31)]
    ident = build_span_csr(byte_bpe, byte_bpe)
    check("identity_byte_pieces", ident.is_identity(),
          f"same offsets did not give the identity: {ident.stats()}")
    for trial in range(20):
        offs = _random_offsets(rng, 300, 6, 0.2)
        csr = build_span_csr(offs, offs)
        if not csr.is_identity():
            check("identity_random", False, f"trial {trial}: {csr.stats()}")
            break
    x = rng.standard_normal((len(byte_bpe), 7)).astype(np.float16)
    check("identity_apply_exact", np.array_equal(ident.apply(x), x),
          "identity pooling changed the rows")

    # Causality over random different tokenizations; and a non-causal variant
    # must be CAUGHT by the independent checker (the leak canary).
    caught = False
    for trial in range(30):
        n_chars = int(rng.integers(50, 400))
        src = _random_offsets(rng, n_chars, 7, 0.15)
        tgt = _random_offsets(rng, n_chars, 4, 0.15)
        csr = build_span_csr(src, tgt)
        bad = causal_violations(csr, src, tgt)
        if bad:
            check("causal_random", False, f"trial {trial}: leaks {bad[:3]}")
            break
        rows = np.diff(csr.indptr)
        if np.any(rows == 0) and csr.n_empty == 0:
            check("empty_rows_counted", False, "an empty row was not counted")
        leaky = build_span_csr(src, tgt, causal=False)
        caught = caught or bool(causal_violations(leaky, src, tgt))
    check("leak_canary_detected", caught,
          "non-causal pooling produced no detectable violation in 30 trials — "
          "the checker cannot see a leak")

    # A straddling source token is never pooled into the earlier target token.
    src = [(0, 6)]                     # "abcdef" as one source token
    tgt = [(0, 3), (3, 6)]             # "abc" "def"
    csr = build_span_csr(src, tgt)
    check("straddle_not_in_first", list(csr.indices[csr.indptr[0]:csr.indptr[1]]) == [],
          "the whole-word source token leaked into the first half")
    check("straddle_in_second", list(csr.indices[csr.indptr[1]:csr.indptr[2]]) == [0],
          f"got {csr.indices.tolist()}")
    check("straddle_leading_empty_counted", csr.n_empty == 1, f"{csr.stats()}")

    # Carry fallback: target splits a character the source keeps whole.
    src = [(0, 2), (2, 3)]
    tgt = [(0, 2), (2, 3), (2, 3)]
    csr = build_span_csr(src, tgt)
    dense = csr.to_dense()
    check("carry_counted", csr.n_carried == 1, f"{csr.stats()}")
    check("carry_copies_previous", np.array_equal(dense[1], dense[0]),
          f"row 1 {dense[1]} is not row 0 {dense[0]}")
    check("carry_last_row", dense[2].tolist() == [0.0, 1.0], f"got {dense[2]}")

    # Mean pooling: two source tokens inside one target token average.
    csr = build_span_csr([(0, 2), (2, 4)], [(0, 4)])
    check("mean_pool", csr.to_dense().tolist() == [[0.5, 0.5]],
          f"got {csr.to_dense().tolist()}")

    # Spec digest is stable and sensitive to every field.
    spec = AlignmentSpec.span("a" * 64, "b" * 64)
    check("digest_stable", spec.digest() == AlignmentSpec.from_dict(spec.to_dict()).digest())
    for field_name, value in (("target_tokenizer_sha256", "c" * 64), ("causal", False),
                              ("rule_version", SPAN_RULE_VERSION + 1)):
        other = AlignmentSpec(**{**spec.to_dict(), field_name: value})
        check(f"digest_covers_{field_name}", other.digest() != spec.digest(),
              f"changing {field_name} left the digest unchanged")
    try:
        AlignmentSpec(method="nearest", source_tokenizer_sha256="a",
                      target_tokenizer_sha256="b")
        check("closed_enum", False, "accepted an unknown alignment method")
    except ValueError as exc:
        check("closed_enum_reason", "not in" in str(exc), str(exc))

    # Window cut by characters.
    src = [(0, 4), (4, 8), (8, 12), (12, 16)]
    tgt = [(0, 2), (2, 6), (6, 10), (10, 14), (14, 16)]
    check("window_cut", source_window(src, tgt, 1, 3) == (0, 2),
          f"got {source_window(src, tgt, 1, 3)}")

    class _Slow:
        is_fast = False

    try:
        require_fast_tokenizer(_Slow(), "source")
        check("slow_refused", False, "accepted a slow tokenizer")
    except ValueError as exc:
        check("slow_reason", "FAST tokenizer" in str(exc), str(exc))

    for f in failures:
        print(f"FAIL {f}")
    print(f"align self-test: {'PASS' if not failures else 'FAIL'} ({len(failures)} failures)")
    return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Character-span alignment for KV transfer")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test()
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
