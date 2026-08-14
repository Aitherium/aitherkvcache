"""Cross-model KV cache transfer.

Convert one model's KV cache into another model's, so the receiving model can
skip prefill entirely. Per-(target layer, head) affine maps fitted by ridge
regression, with source layers chosen per target layer by held-out R².

    capture  ->  fit  ->  measure  ->  translate

Each stage is a CLI as well as an API::

    python -m aither_kvcache.kvtransfer.capture  --source A --target B --out CAP
    python -m aither_kvcache.kvtransfer.fit      --capture CAP --out PACK
    python -m aither_kvcache.kvtransfer.transfer --pack PACK --source A --target B --record

**The measure step is not optional.** ``load_pack`` refuses any pack that does
not carry downstream acceptance evidence over a large enough sample, because a
converted cache fails silently: an approximate one does not raise, it makes the
receiving model produce fluent, on-topic, confidently wrong text with a green
healthcheck and nothing in a log.

Two preconditions are definitional rather than quality knobs. The two models
must **tokenize identically** — the map sends position *i* to position *i*, so
different tokenizers mean the fit is regressing misaligned rows — and the RoPE
schedule must be exactly reproducible. Both are enforced.

Only ``numpy`` is needed for the mapper, the pack format and the rotation math.
``capture`` and ``transfer``'s evaluation additionally need ``torch`` and
``transformers``; they are imported lazily so the rest of the package works
without them.
"""

from __future__ import annotations

from .geometry import KVGeometry, PairVerdict, check_pair, source_roles_for
from .mapper import GramStats, LayerMap, apply_layer_map, fit_layer_map, r2_from_gram
from .pack import (
    AcceptanceMetrics,
    LoadedPack,
    PackManifest,
    PackRefusedError,
    load_pack,
    record_acceptance,
    write_pack,
)
from .rope import RopeSpec, apply_rope, rotate_to_target, strip_rope

__all__ = [
    "AcceptanceMetrics",
    "GramStats",
    "KVGeometry",
    "LayerMap",
    "LoadedPack",
    "PackManifest",
    "PackRefusedError",
    "PairVerdict",
    "RopeSpec",
    "apply_layer_map",
    "apply_rope",
    "check_pair",
    "fit_layer_map",
    "load_pack",
    "r2_from_gram",
    "record_acceptance",
    "rotate_to_target",
    "source_roles_for",
    "strip_rope",
    "write_pack",
]

__version__ = "2.4.0"
