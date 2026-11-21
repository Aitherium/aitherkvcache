"""Residual-stream state bridges (LatentPort method) for attention-only Qwen3-family pairs.

A small SOURCE model reads the prompt; a fitted bridge translates its residual
streams into the key/value cache a larger TARGET model would have built; the
target reads ONE bridge token (the prompt's last) and answers. The target never
prefills the prompt history.

Method credit: LatentPort, Simon P. Villani (arXiv:2609.25053, latentport.com,
github.com/LatentPort/latentport, Apache-2.0) -- the per-layer closed-form ridge
maps, the one-token handoff and the rank-4 learned correction ``X' = X(I + B U^T)``
trained through the frozen receiver. The unmatched-geometry variant (map from the
source's RESIDUAL stream entering the proportionally placed layer, two taps, keys
de-rotated before fitting) and the bridge file format follow PrattleNot by
agentic-underground (github.com/agentic-underground/PrattleNot, Apache-2.0), whose
published ``qwen3-0.6b-to-1.7b`` bridge this module reads unchanged. This is a
clean Python/torch port of those published specifications; no code is vendored.

Why this and not ``mapper.py``: the older lane maps the source's K/V to the
target's K/V. PrattleNot measured that failing across a head-count change
(held-out NCR -0.31) and our own SmolLM2 pack sat +1.83 nats above native
(README.md here). The residual stream carries what the target's heads need.

Bridge file (safetensors, PrattleNot ``KvMap``):
  meta      u32 [src_heads, tgt_heads, dim, taps]
  layer_of  u32 [target_layers]       source layer feeding each target layer
  k.<l>     f16 [taps*hidden_src + 1, tgt_heads*dim]   (target keys DE-ROTATED)
  v.<l>     f16 [taps*hidden_src + 1, tgt_heads*dim]
  ck.<l>    f32 [dim, dim]  optional folded correction I + B U^T (keys, post-RoPE)
  cv.<l>    f32 [dim, dim]  optional (values)

torch/transformers/safetensors are imported lazily; ``_self_test`` runs on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

RANK = 4
POSITIONS = 9
IDENTITY_WEIGHT = 0.01


def layer_of(li: int, source_layers: int, target_layers: int) -> int:
    """Target layer ``l`` reads source layer round(l*(S-1)/(T-1))."""
    if target_layers < 2:
        return 0
    return int(round(li * (source_layers - 1) / (target_layers - 1)))


def qwen3_prompt(user: str) -> str:
    """Qwen3's turn with thinking off -- PrattleNot's Qwen3 bridges' template."""
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _rotate_half(x):
    import torch

    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope(k, cos, sin, inverse: bool = False):
    """Apply (or undo) the target's rotary embedding. ``cos``/``sin`` are ``[1, n, dim]`` as the
    model's ``rotary_emb`` returns them -- they may carry an attention scale (YaRN), so the inverse
    divides by ``cos^2 + sin^2`` rather than assuming a pure rotation."""
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    if not inverse:
        return k * c + _rotate_half(k) * s
    return (k * c - _rotate_half(k) * s) / (c * c + s * s)


@dataclass
class KvMap:
    layer_of: List[int]
    taps: int
    src_heads: int
    tgt_heads: int
    dim: int
    k: list
    v: list
    ck: list = field(default_factory=list)
    cv: list = field(default_factory=list)

    @classmethod
    def load(cls, path: str, device=None) -> "KvMap":
        import torch
        from safetensors.torch import load_file

        t = load_file(path)
        meta = [int(x) for x in t["meta"].tolist()]
        lo = [int(x) for x in t["layer_of"].tolist()]
        f = lambda n: t[n].to(device=device, dtype=torch.float32)  # noqa: E731
        k = [f(f"k.{i}") for i in range(len(lo))]
        v = [f(f"v.{i}") for i in range(len(lo))]
        ck = [f(f"ck.{i}") for i in range(len(lo))] if "ck.0" in t else []
        cv = [f(f"cv.{i}") for i in range(len(lo))] if "cv.0" in t else []
        return cls(lo, meta[3] if len(meta) > 3 else 1, meta[0], meta[1], meta[2], k, v, ck, cv)

    def save(self, path: str) -> None:
        import torch
        from safetensors.torch import save_file

        m: Dict[str, "torch.Tensor"] = {
            "meta": torch.tensor(
                [self.src_heads, self.tgt_heads, self.dim, self.taps], dtype=torch.int32
            ),
            "layer_of": torch.tensor(self.layer_of, dtype=torch.int32),
        }
        for i, (k, v) in enumerate(zip(self.k, self.v)):
            m[f"k.{i}"] = k.detach().to("cpu", torch.float16).contiguous()
            m[f"v.{i}"] = v.detach().to("cpu", torch.float16).contiguous()
        for i, (k, v) in enumerate(zip(self.ck, self.cv)):
            m[f"ck.{i}"] = k.detach().to("cpu", torch.float32).contiguous()
            m[f"cv.{i}"] = v.detach().to("cpu", torch.float32).contiguous()
        # PrattleNot stores meta/layer_of as u32; safetensors-torch has no uint32 writer on every
        # version, so int32 is written and both readers accept it (values are small and positive).
        save_file(m, path)


def tapped_features(hidden: Sequence, first: int, taps: int):
    """``taps`` consecutive source residual streams from ``first`` (clamped), plus an intercept."""
    import torch

    last = len(hidden) - 1
    parts = [hidden[min(first + t, last)] for t in range(taps)]
    x = torch.cat(parts, dim=-1)[0].float()  # [n, taps*hidden]
    return torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=1)


def capture(model, ids):
    """The source's INPUT residual stream to every layer, ``[1, n, hidden]`` each."""
    out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return list(out.hidden_states[: model.config.num_hidden_layers])


def target_rope(target, n: int, device, offset: int = 0):
    import torch

    pos = torch.arange(offset, offset + n, device=device)[None]
    dummy = torch.zeros(1, n, target.config.head_dim, device=device, dtype=torch.float32)
    cos, sin = target.model.rotary_emb(dummy, pos)
    return cos.float(), sin.float()


def translate(kmap: KvMap, hidden: Sequence, target, corrections=None) -> List[tuple]:
    """Per target layer ``(K, V)`` as ``[1, tgt_heads, n, dim]`` float32 (K rotated).

    ``corrections`` (training only) overrides the folded ck/cv with live ``(dk, dv)`` callables."""
    n = hidden[0].shape[1]
    cos, sin = target_rope(target, n, hidden[0].device)
    out = []
    for li, s in enumerate(kmap.layer_of):
        f = tapped_features(hidden, s, kmap.taps)
        heads = lambda y: y.reshape(1, n, kmap.tgt_heads, kmap.dim).transpose(1, 2)  # noqa: E731
        k = rope(heads(f @ kmap.k[li]), cos, sin)
        v = heads(f @ kmap.v[li])
        if corrections is not None:
            k, v = corrections(li, k, v)
        elif kmap.ck:
            k, v = k @ kmap.ck[li], v @ kmap.cv[li]
        out.append((k, v))
    return out


def as_cache(layers: List[tuple], target):
    from transformers import DynamicCache

    cache = DynamicCache(config=target.config)
    dt = next(target.parameters()).dtype
    for i, (k, v) in enumerate(layers):
        cache.update(k.to(dt), v.to(dt), i)
    return cache


def step_from(target, cache, ids, start: int, all_logits: bool = False):
    """Run ``ids`` ([1, m]) on ``target`` from absolute position ``start`` over ``cache``."""
    import torch

    m = ids.shape[1]
    past = cache.get_seq_length() if cache is not None else 0
    pos = torch.arange(start, start + m, device=ids.device)[None]
    cpos = torch.arange(past, past + m, device=ids.device)
    out = target(
        input_ids=ids,
        past_key_values=cache,
        position_ids=pos,
        cache_position=cpos,
        use_cache=True,
        logits_to_keep=0 if all_logits else 1,
    )
    return out.logits[0] if all_logits else out.logits[0, -1]


# ---------------------------------------------------------------------------- fitting


def greedy_reply(model, tok, prompt: str, n_tokens: int, device) -> List[int]:
    import torch

    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device=device)
    out = model.generate(
        ids, max_new_tokens=n_tokens, do_sample=False, temperature=None, top_p=None, top_k=None
    )
    return out[0, ids.shape[1] :].tolist()


def fit(
    source,
    target,
    tok,
    questions: Sequence[str],
    *,
    taps: int = 2,
    lam: float = 1e-3,
    reply_tokens: int = 48,
    layer_group: int = 12,
    log=print,
) -> KvMap:
    """Closed-form ridge maps, one per target layer and role, from the source's residual stream.

    Rows are every position of ``prompt + target's greedy reply`` (PrattleNot's bridge-fit). Sums
    are float64 on the target's device, in groups of target layers so the Gram matrices fit
    beside both models."""
    import torch

    dev = next(target.parameters()).device
    sl, tl = source.config.num_hidden_layers, target.config.num_hidden_layers
    th, dim = target.config.num_key_value_heads, target.config.head_dim
    hidden = source.config.hidden_size
    d_in, d_out = taps * hidden + 1, th * dim
    lo = [layer_of(li, sl, tl) for li in range(tl)]

    rows_ids = []
    for i, q in enumerate(questions):
        p = qwen3_prompt(q)
        reply = greedy_reply(target, tok, p, reply_tokens, dev)
        rows_ids.append(tok.encode(p, add_special_tokens=False) + reply)
        if (i + 1) % 50 == 0:
            log(f"replied {i + 1}/{len(questions)}")

    ks: List = [None] * tl
    vs: List = [None] * tl
    for g0 in range(0, tl, layer_group):
        group = list(range(g0, min(tl, g0 + layer_group)))
        srcs = sorted({lo[li] for li in group})
        xtx = {s: torch.zeros(d_in, d_in, dtype=torch.float64, device=dev) for s in srcs}
        xty = {li: torch.zeros(d_in, 2 * d_out, dtype=torch.float64, device=dev) for li in group}
        with torch.no_grad():
            for ids in rows_ids:
                t = torch.tensor([ids], device=dev)
                hs = capture(source, t)
                out = target(input_ids=t, use_cache=True, logits_to_keep=1)
                cos, sin = target_rope(target, t.shape[1], dev)
                feats = {s: tapped_features(hs, s, taps).double() for s in srcs}
                for s in srcs:
                    xtx[s] += feats[s].T @ feats[s]
                for li in group:
                    kt, vt = (
                        out.past_key_values.layers[li].keys,
                        out.past_key_values.layers[li].values,
                    )
                    yk = (
                        rope(kt.float(), cos, sin, inverse=True)[0]
                        .transpose(0, 1)
                        .reshape(-1, d_out)
                    )
                    yv = vt.float()[0].transpose(0, 1).reshape(-1, d_out)
                    xty[li] += feats[lo[li]].T @ torch.cat([yk, yv], dim=1).double()
                del out, hs, feats
            for s in srcs:
                a = xtx[s]
                a = a + lam * a.diagonal().mean().clamp_min(1e-12) * torch.eye(
                    d_in, dtype=a.dtype, device=dev
                )
                chol = torch.linalg.cholesky(a)
                for li in group:
                    if lo[li] != s:
                        continue
                    w = torch.cholesky_solve(xty[li], chol).float()
                    ks[li], vs[li] = w[:, :d_out].contiguous(), w[:, d_out:].contiguous()
            del xtx, xty
            torch.cuda.empty_cache()
        log(f"fitted target layers {group[0]}..{group[-1]}")
    return KvMap(lo, taps, source.config.num_key_value_heads, th, dim, ks, vs)


def _basis(dim: int, seed: int, device):
    """A fixed orthonormal [dim, RANK] basis, per layer and role (PrattleNot's xorshift)."""
    import torch

    mask = (1 << 64) - 1
    state = (seed * 0x9E3779B97F4A7C15 & mask) | 1
    cols: List[List[float]] = []
    while len(cols) < RANK:
        v = []
        for _ in range(dim):
            state ^= (state << 13) & mask
            state ^= state >> 7
            state ^= (state << 17) & mask
            v.append((state % 2001) / 1000.0 - 1.0)
        for c in cols:
            d = sum(a * b for a, b in zip(v, c))
            v = [a - d * b for a, b in zip(v, c)]
        nrm = math.sqrt(sum(a * a for a in v))
        cols.append([a / nrm for a in v])
    return torch.tensor(
        [[c[i] for c in cols] for i in range(dim)], dtype=torch.float32, device=device
    )


def correct(
    kmap: KvMap,
    source,
    target,
    tok,
    questions: Sequence[str],
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    accumulate: int = 8,
    log=print,
) -> KvMap:
    """The paper's rank-4 residual correction, trained through the FROZEN target against the native
    target's distribution at the bridge token and 8 teacher-forced continuation tokens."""
    import torch

    dev = next(target.parameters()).device
    for p in list(target.parameters()) + list(source.parameters()):
        p.requires_grad_(False)
    kmap.ck, kmap.cv = [], []
    n_layers, dim = len(kmap.layer_of), kmap.dim
    data = []
    with torch.no_grad():
        for q in questions:
            ids = tok.encode(qwen3_prompt(q), add_special_tokens=False)
            n = len(ids)
            ref = greedy_reply(target, tok, qwen3_prompt(q), POSITIONS - 1, dev)
            if len(ref) < POSITIONS - 1:
                continue
            allids = torch.tensor([ids + ref], device=dev)
            logits = (
                target(input_ids=allids, use_cache=False)
                .logits[0, n - 1 : n - 1 + POSITIONS]
                .float()
            )
            native = torch.softmax(logits, -1).to(torch.bfloat16).cpu()
            hs = capture(source, torch.tensor([ids[:-1]], device=dev))
            state = [(k.cpu(), v.cpu()) for k, v in translate(kmap, hs, target)]
            data.append((ids, ref, native, state))
    log(f"prepared {len(data)} correction examples")
    bases = [(_basis(dim, 2 * li + 1, dev), _basis(dim, 2 * li + 2, dev)) for li in range(n_layers)]
    us = [
        (
            torch.zeros(dim, RANK, device=dev, requires_grad=True),
            torch.zeros(dim, RANK, device=dev, requires_grad=True),
        )
        for _ in range(n_layers)
    ]
    opt = torch.optim.AdamW(
        [u for pair in us for u in pair], lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )
    for ep in range(epochs):
        total = 0.0
        opt.zero_grad(set_to_none=True)
        for i, (ids, ref, native, state) in enumerate(data):
            n = len(ids)
            pen = torch.zeros((), device=dev)
            layers = []
            for li, (k, v) in enumerate(state):
                k, v = k.to(dev), v.to(dev)
                (bk, bv), (uk, uv) = bases[li], us[li]
                dk, dv = (k @ bk) @ uk.T, (v @ bv) @ uv.T
                pen = (
                    pen
                    + (dk.square().sum() + 1e-12).sqrt() / k.norm()
                    + (dv.square().sum() + 1e-12).sqrt() / v.norm()
                )
                layers.append((k + dk, v + dv))
            cache = as_cache(layers, target)
            inp = torch.tensor([[ids[-1]] + ref], device=dev)
            logits = step_from(target, cache, inp, n - 1, all_logits=True).float()
            logq = torch.log_softmax(logits, -1)
            p = native.to(dev).float()
            kl = (p * (p.clamp_min(1e-12).log() - logq)).sum() / POSITIONS
            loss = kl + IDENTITY_WEIGHT * pen / n_layers
            (loss / accumulate).backward()
            total += float(loss.detach())
            if (i + 1) % accumulate == 0 or i + 1 == len(data):
                opt.step()
                opt.zero_grad(set_to_none=True)
        log(f"correction epoch {ep + 1}: mean loss {total / max(1, len(data)):.4f}")
    eye = torch.eye(dim, device=dev)
    kmap.ck = [(eye + bk @ uk.detach().T) for (bk, _), (uk, _) in zip(bases, us)]
    kmap.cv = [(eye + bv @ uv.detach().T) for (_, bv), (_, uv) in zip(bases, us)]
    return kmap


# ---------------------------------------------------------------------------- measuring


def ncr(empty: float, condition: float, native: float) -> float:
    """Native-closure ratio (arXiv:2609.25053 Eq. 8): share of the empty -> native gap."""
    return (empty - condition) / (empty - native)


def fidelity(
    kmap: KvMap, source, target, tok, questions: Sequence[str], ref_tokens: int = 24
) -> Dict[str, float]:
    """Held-out NLL/token of the target's own greedy reply, teacher-forced, 4 conditions."""
    import torch

    dev = next(target.parameters()).device
    sums = {"native": 0.0, "bridged": 0.0, "empty": 0.0, "wrong_donor": 0.0, "source_alone": 0.0}
    ids_all = [tok.encode(qwen3_prompt(q), add_special_tokens=False) for q in questions]

    def score(cache, first_ids, start, ref):
        inp = torch.tensor([first_ids + ref[:-1]], device=dev)
        logits = step_from(target, cache, inp, start, all_logits=True).float()
        lp = torch.log_softmax(logits[len(first_ids) - 1 :], -1)
        return float(-lp[torch.arange(len(ref)), torch.tensor(ref, device=dev)].mean())

    with torch.no_grad():
        for i, ids in enumerate(ids_all):
            n = len(ids)
            ref = greedy_reply(target, tok, qwen3_prompt(questions[i]), ref_tokens, dev)
            sums["native"] += score(None, ids, 0, ref)
            st = as_cache(
                translate(kmap, capture(source, torch.tensor([ids[:-1]], device=dev)), target),
                target,
            )
            sums["bridged"] += score(st, [ids[-1]], n - 1, ref)
            sums["empty"] += score(None, [ids[-1]], n - 1, ref)
            j = (i + 1) % len(ids_all)
            dn = len(ids_all[j])
            st = as_cache(
                translate(
                    kmap, capture(source, torch.tensor([ids_all[j][:-1]], device=dev)), target
                ),
                target,
            )
            sums["wrong_donor"] += score(st, [ids[-1]], dn - 1, ref)
    k = float(len(ids_all))
    r = {key: val / k for key, val in sums.items() if key != "source_alone"}
    r["ncr_bridged"] = ncr(r["empty"], r["bridged"], r["native"])
    r["ncr_wrong_donor"] = ncr(r["empty"], r["wrong_donor"], r["native"])
    return r


# ---------------------------------------------------------------------------- self-test


def _self_test() -> int:
    import torch

    torch.manual_seed(0)
    fails = 0
    k = torch.randn(1, 2, 5, 8)
    ang = torch.rand(1, 5, 4) * 3
    cos = torch.cat([ang.cos(), ang.cos()], -1) * 1.14
    sin = torch.cat([ang.sin(), ang.sin()], -1) * 1.14
    back = rope(rope(k, cos, sin), cos, sin, inverse=True)
    if float((back - k).abs().max()) > 1e-4:
        print("FAIL rope inverse")
        fails += 1
    if [layer_of(li, 28, 36) for li in (0, 35)] != [0, 27]:
        print("FAIL layer_of ends")
        fails += 1
    b = _basis(16, 3, "cpu")
    if float((b.T @ b - torch.eye(RANK)).abs().max()) > 1e-4:
        print("FAIL basis orthonormal")
        fails += 1
    if abs(ncr(2.0, 1.0, 0.0) - 0.5) > 1e-12:
        print("FAIL ncr")
        fails += 1
    print("residual_bridge self-test:", "OK" if not fails else f"{fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
