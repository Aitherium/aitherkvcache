"""
Per-request tenant isolation for the aither-kvcache vLLM plugin.

This is the SELF-CONTAINED, in-wheel port of AitherOS's per-request KV-cache
tenant isolation. It rides the ACTIVE path on the orchestrator: the aither-kvcache
`vllm.general_plugins` entrypoint (`plugin.register()`) loads in every vLLM
process (API server + EngineCore workers), so a hook installed here actually runs
— unlike the AitherOS `lib.gpu.turboquant` import hooks, which are not mounted
into the served image.

Three seams, all fail-closed and non-fatal (a wrong symbol degrades to the
instance tenant; it never crashes vLLM startup):

  1. Serving layer (API-server process): read the gateway-signed OpenAI ``user``
     field, verify its HMAC, and publish the tenant into a ContextVar.
  2. Engine ``generate`` (API-server process): embed the verified tenant into the
     real vLLM ``request_id`` (positional index 2 in vLLM 0.19.x), so it crosses
     the frontend -> EngineCore process boundary verbatim.
  3. Block manager (EngineCore worker): on ``KVCacheManager.allocate_slots`` the
     ``request.request_id`` carries the embedded tenant — tag the request's blocks
     with it; untag on ``free``.

Self-contained by necessity: the wheel cannot import ``lib.gpu.turboquant``. The
crypto/transport helpers mirror ``request_tenant_registry.py`` byte-for-byte so
tokens the AitherOS gateway signs verify here.

Gate: AITHER_REQUEST_TENANT_ISOLATION=1 (default OFF). Key: AITHER_TENANT_SIGNING_KEY
(injected from AitherSecrets into the image env; empty => fail-closed).
"""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import logging
import os
import sys
import threading
from typing import Any

logger = logging.getLogger("aither.kvcache.tenant")

_ENABLED = os.environ.get("AITHER_REQUEST_TENANT_ISOLATION", "0") == "1"

# Must match request_tenant_registry.py exactly (cross-repo token compatibility).
_TOKEN_SEP = "|"
_RID_PREFIX = "tqt1:"
_RID_SEP = "~"

_pending_tenant: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tq_pending_tenant", default=None
)

# block_id -> tenant_slug. Populated at allocate_slots, cleared at free. Guarded.
_block_tenant: dict[int, str] = {}
_block_lock = threading.Lock()

_isolation_status: dict[str, bool] = {
    "enabled": _ENABLED,
    "serving_patched": False,
    "engine_patched": False,
    "block_manager_patched": False,
    "key_present": False,
    "verified": False,
}


# ---------------------------------------------------------------------------
# Token crypto + request_id transport (mirror of request_tenant_registry.py)
# ---------------------------------------------------------------------------
def _signing_key() -> bytes:
    return os.environ.get("AITHER_TENANT_SIGNING_KEY", "").encode("utf-8")


def verify_tenant_token(
    token: str | None, *, now: float | None = None
) -> str | None:
    """Return the tenant slug iff ``token`` has a valid, unexpired gateway signature."""
    import time
    key = _signing_key()
    if not token or not key:
        return None
    parts = token.split(_TOKEN_SEP)
    if len(parts) != 3:
        return None
    tenant, exp_str, sig = parts
    if not tenant or not exp_str or not sig:
        return None
    try:
        exp = int(exp_str)
    except (TypeError, ValueError):
        return None
    expected = hmac.new(
        key, f"{tenant}{_TOKEN_SEP}{exp}".encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        logger.warning("Rejected tenant token with invalid signature")
        return None
    if exp < (now if now is not None else time.time()):
        logger.warning("Rejected expired tenant token")
        return None
    return tenant


def embed_tenant_in_request_id(request_id: str, tenant_slug: str) -> str:
    """Prefix a verified tenant onto a request_id (idempotent, boundary-safe)."""
    if not request_id or not tenant_slug:
        return request_id
    rid = str(request_id)
    if _RID_SEP in tenant_slug or rid.startswith(_RID_PREFIX):
        return request_id
    return f"{_RID_PREFIX}{tenant_slug}{_RID_SEP}{rid}"


def extract_tenant_from_request_id(
    request_id: str | None,
) -> tuple[str | None, str | None]:
    """Inverse of embed_tenant_in_request_id -> (tenant_or_None, original_id)."""
    if not request_id:
        return None, request_id
    rid = str(request_id)
    if not rid.startswith(_RID_PREFIX):
        return None, request_id
    tenant, sep, original = rid[len(_RID_PREFIX):].partition(_RID_SEP)
    if not sep or not tenant:
        return None, request_id
    return tenant, original


# ---------------------------------------------------------------------------
# Block -> tenant map
# ---------------------------------------------------------------------------
def tag_blocks(tenant_slug: str, block_ids) -> None:
    if not tenant_slug or not block_ids:
        return
    with _block_lock:
        for bid in block_ids:
            _block_tenant[int(bid)] = tenant_slug


def untag_blocks(block_ids) -> None:
    if not block_ids:
        return
    with _block_lock:
        for bid in block_ids:
            _block_tenant.pop(int(bid), None)


def block_tenant(block_id: int) -> str | None:
    with _block_lock:
        return _block_tenant.get(int(block_id))


def block_tenant_count() -> int:
    with _block_lock:
        return len(_block_tenant)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def get_isolation_status() -> dict[str, bool]:
    _isolation_status["enabled"] = _ENABLED
    _isolation_status["key_present"] = bool(_signing_key())
    _isolation_status["verified"] = (
        _isolation_status["enabled"]
        and _isolation_status["serving_patched"]
        and _isolation_status["engine_patched"]
        and _isolation_status["block_manager_patched"]
        and _isolation_status["key_present"]
    )
    return dict(_isolation_status)


# ---------------------------------------------------------------------------
# Seam 1: serving-layer wrap (verified tenant -> ContextVar)
# ---------------------------------------------------------------------------
def _wrap_serving_method(cls: Any, method_name: str) -> bool:
    orig = getattr(cls, method_name, None)
    if orig is None:
        return False
    if getattr(orig, "_tq_tenant_wrapped", False):
        return True

    async def _tenant_serving_wrapper(self, request, *args, **kwargs):
        token = None
        try:
            user = getattr(request, "user", None)
            tenant = verify_tenant_token(user) if user else None
            if tenant:
                token = _pending_tenant.set(tenant)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("tenant capture skipped: %s", e)
            token = None
        try:
            return await orig(self, request, *args, **kwargs)
        finally:
            if token is not None:
                _pending_tenant.reset(token)

    _tenant_serving_wrapper._tq_tenant_wrapped = True
    setattr(cls, method_name, _tenant_serving_wrapper)
    return True


# ---------------------------------------------------------------------------
# Seam 2: engine generate wrap (embed tenant into positional request_id)
# ---------------------------------------------------------------------------
# vLLM 0.19.x: AsyncLLM.generate(self, prompt, sampling_params, request_id, *, ...)
# request_id is the 3rd positional arg -> index 2 in the wrapper's *args.
_GENERATE_RID_POS = 2


def _wrap_engine_generate(cls: Any, rid_pos: int = _GENERATE_RID_POS) -> bool:
    orig = getattr(cls, "generate", None)
    if orig is None:
        return False
    if getattr(orig, "_tq_tenant_wrapped", False):
        return True

    async def _tenant_generate_wrapper(self, *args, **kwargs):
        tenant = _pending_tenant.get(None)
        if tenant:
            rid = None
            pos = None
            if isinstance(kwargs.get("request_id"), str) and kwargs["request_id"]:
                rid = kwargs["request_id"]
            elif (len(args) > rid_pos and isinstance(args[rid_pos], str)
                  and args[rid_pos]):
                pos = rid_pos
                rid = args[pos]
            if rid:
                embedded = embed_tenant_in_request_id(rid, tenant)
                if pos is not None:
                    args = args[:pos] + (embedded,) + args[pos + 1:]
                else:
                    kwargs = dict(kwargs)
                    kwargs["request_id"] = embedded
        async for out in orig(self, *args, **kwargs):
            yield out

    _tenant_generate_wrapper._tq_tenant_wrapped = True
    cls.generate = _tenant_generate_wrapper
    return True


# ---------------------------------------------------------------------------
# Seam 3: block manager wrap (tag blocks by tenant from request.request_id)
# ---------------------------------------------------------------------------
def _wrap_block_manager(cls: Any) -> bool:
    alloc = getattr(cls, "allocate_slots", None)
    free = getattr(cls, "free", None)
    if alloc is None:
        return False
    if getattr(alloc, "_tq_tenant_wrapped", False):
        return True

    def _flatten_block_ids(mgr, request_id) -> list:
        try:
            groups = mgr.get_block_ids(request_id)
        except Exception:
            return []
        ids: list = []
        for g in groups or ():
            try:
                ids.extend(int(b) for b in g)
            except TypeError:
                ids.append(int(g))
        return ids

    def _tenant_allocate_slots(self, request, *args, **kwargs):
        result = alloc(self, request, *args, **kwargs)
        if result is not None:
            try:
                rid = getattr(request, "request_id", None)
                tenant, _orig = extract_tenant_from_request_id(rid)
                if tenant:
                    tag_blocks(tenant, _flatten_block_ids(self, rid))
            except Exception as e:  # never break allocation for tagging
                logger.debug("tenant block-tag skipped: %s", e)
        return result

    _tenant_allocate_slots._tq_tenant_wrapped = True
    cls.allocate_slots = _tenant_allocate_slots

    if free is not None and not getattr(free, "_tq_tenant_wrapped", False):
        def _tenant_free(self, request, *args, **kwargs):
            try:
                rid = getattr(request, "request_id", None)
                if rid:
                    untag_blocks(_flatten_block_ids(self, rid))
            except Exception as e:
                logger.debug("tenant block-untag skipped: %s", e)
            return free(self, request, *args, **kwargs)

        _tenant_free._tq_tenant_wrapped = True
        cls.free = _tenant_free

    return True


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
def install_tenant_isolation() -> bool:
    """Install all three seams. Called from plugin.register() when enabled.

    Returns True iff verified (serving + engine + block manager + key). Every
    failure is non-fatal and leaves the process serving with the instance tenant.
    """
    if not _ENABLED:
        logger.debug("tenant isolation disabled (AITHER_REQUEST_TENANT_ISOLATION!=1)")
        return False

    serving_targets = [
        ("vllm.entrypoints.openai.chat_completion.serving", "OpenAIServingChat",
         "create_chat_completion"),
        ("vllm.entrypoints.openai.serving_chat", "OpenAIServingChat",
         "create_chat_completion"),
        ("vllm.entrypoints.openai.completion.serving", "OpenAIServingCompletion",
         "create_completion"),
        ("vllm.entrypoints.openai.serving_completion", "OpenAIServingCompletion",
         "create_completion"),
    ]
    seen = set()
    for modpath, clsname, method in serving_targets:
        try:
            mod = __import__(modpath, fromlist=[clsname])
            cls = getattr(mod, clsname)
            if cls in seen:
                continue
            if _wrap_serving_method(cls, method):
                seen.add(cls)
                _isolation_status["serving_patched"] = True
        except Exception as e:
            logger.debug("serving wrap skipped %s.%s: %s", clsname, method, e)

    for modpath, clsname in [
        ("vllm.v1.engine.async_llm", "AsyncLLM"),
        ("vllm.engine.async_llm_engine", "AsyncLLMEngine"),
    ]:
        try:
            mod = __import__(modpath, fromlist=[clsname])
            if _wrap_engine_generate(getattr(mod, clsname)):
                _isolation_status["engine_patched"] = True
        except Exception as e:
            logger.debug("engine wrap skipped %s: %s", clsname, e)

    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        if _wrap_block_manager(KVCacheManager):
            _isolation_status["block_manager_patched"] = True
    except Exception as e:
        logger.debug("block-manager wrap skipped: %s", e)

    st = get_isolation_status()
    print(
        f"[aither-kvcache] tenant isolation: serving={st['serving_patched']} "
        f"engine={st['engine_patched']} block_mgr={st['block_manager_patched']} "
        f"key={st['key_present']} VERIFIED={st['verified']}",
        file=sys.stderr, flush=True,
    )
    return st["verified"]
