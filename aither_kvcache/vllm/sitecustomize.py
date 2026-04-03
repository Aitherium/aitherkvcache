"""
TurboQuant site-customize hook -- loaded in ALL vLLM processes.

Registers TurboQuant as a custom vLLM attention backend via the official
plugin system (register_backend). No monkey-patching.

Three-phase hook:
  Phase 1 (registry import): register CUSTOM backend
  Phase 2 (kv_cache_interface import): apply capacity + block manager patches
  Phase 3 (gpu_model_runner import): apply reshape patch (PRIMARY mode only)
  Separated because each module is not available during earlier phases.

Phase 3 is needed because the reshape patch targets GPUModelRunner, which
lives in vllm.v1.worker.gpu_model_runner.  In PRIMARY mode, apply_tq_patches()
(Phase 2) attempts to import and patch GPUModelRunner directly.  If that
import hasn't happened yet (vLLM imports modules lazily in worker processes),
Phase 3 catches it when the module is finally loaded.

Usage:
  Copy this file to your Python path as sitecustomize.py:
    cp $(python -c "import aither_kvcache.vllm.sitecustomize; print(turboquant.vllm.sitecustomize.__file__)") /path/to/sitecustomize.py
    export PYTHONPATH="/path/to:$PYTHONPATH"

Activated via: PYTHONPATH=/path/to AITHER_TQ_BITS=4 vllm serve ...
"""
import os
import sys

_TQ_BITS = int(os.environ.get("AITHER_TQ_BITS", "0"))
# Derive primary flag from AITHER_TQ_MODE (canonical) or AITHER_TQ_PRIMARY (legacy)
_TQ_MODE = os.environ.get("AITHER_TQ_MODE", "")
_TQ_PRIMARY = _TQ_MODE.endswith("-primary") if _TQ_MODE else (
    os.environ.get("AITHER_TQ_PRIMARY", "0") == "1"
)

if _TQ_BITS in (2, 3, 4):
    # Three-phase import hook:
    #   Phase 1: register backend (on registry import)
    #   Phase 2: apply engine patches (on kv_cache_interface import)
    #   Phase 3: apply reshape patch if missed in Phase 2 (on gpu_model_runner
    #            import, PRIMARY mode only)
    import builtins
    _original_import = builtins.__import__
    _backend_registered = False
    _patches_applied = False
    _reshape_ensured = False

    def _tq_import_hook(name, *args, **kwargs):
        global _backend_registered, _patches_applied, _reshape_ensured
        mod = _original_import(name, *args, **kwargs)

        # Phase 1: Apply TQ hooks to TritonAttentionImpl (PRIMARY mode)
        # or register custom backend (SHADOW mode, legacy).
        # Hook approach: monkey-patch TritonAttentionImpl.forward instead of
        # registering as CUSTOM backend. This preserves torch.compile compat.
        if not _backend_registered and name == "vllm.v1.attention.backends.triton_attn":
            _backend_registered = True
            try:
                if _TQ_PRIMARY:
                    from aither_kvcache.vllm.hooks import apply_tq_hooks
                    ok = apply_tq_hooks()
                    pid = os.getpid()
                    status = "OK" if ok else "FAILED"
                    print(f"[TQ] pid={pid}: Hooks applied to TritonAttentionImpl ({status})",
                          file=sys.stderr)
                else:
                    from aither_kvcache.vllm.backend import (
                        register_turboquant_backend,
                    )
                    register_turboquant_backend()
                    pid = os.getpid()
                    print(f"[TQ] pid={pid}: Registered CUSTOM backend (TQ{_TQ_BITS})",
                          file=sys.stderr)
            except Exception as e:
                print(f"[TQ] Phase 1 failed: {e}", file=sys.stderr)

        # Phase 2: apply capacity patches (deferred until kv_cache_interface is loaded)
        if not _patches_applied and name == "vllm.v1.kv_cache_interface":
            _patches_applied = True
            try:
                from aither_kvcache.vllm.engine import apply_tq_patches
                ok = apply_tq_patches(bits=_TQ_BITS)
                pid = os.getpid()
                status = "OK" if ok else "PARTIAL"
                mode = "PRIMARY" if _TQ_PRIMARY else "SHADOW"
                print(f"[TQ] pid={pid}: Engine patches {status} ({mode} mode)",
                      file=sys.stderr)
            except Exception as e:
                pid = os.getpid()
                print(f"[TQ] pid={pid}: Engine patches failed (non-fatal): {e}",
                      file=sys.stderr)

        # Phase 3: ensure reshape patch is applied when GPUModelRunner is loaded.
        # In PRIMARY mode, _patch_reshape() is called during Phase 2 (apply_tq_patches).
        # If GPUModelRunner hadn't been imported yet at that point, the import
        # inside _patch_reshape would trigger it.  But in some vLLM worker
        # process orderings, kv_cache_interface loads BEFORE gpu_model_runner.
        # This phase catches the case where Phase 2's _patch_reshape() succeeded
        # (GPUModelRunner was importable) but also serves as a safety net if
        # the import order ever changes.
        if (_TQ_PRIMARY and not _reshape_ensured
                and name == "vllm.v1.worker.gpu_model_runner"):
            _reshape_ensured = True
            try:
                from vllm.v1.worker.gpu_model_runner import GPUModelRunner
                if not getattr(GPUModelRunner, "_tq_reshape_patched", False):
                    from aither_kvcache.vllm.engine import _patch_reshape
                    ok = _patch_reshape()
                    pid = os.getpid()
                    status = "OK" if ok else "FAILED"
                    print(f"[TQ] pid={pid}: Phase 3 reshape patch {status}",
                          file=sys.stderr)
                else:
                    pid = os.getpid()
                    print(f"[TQ] pid={pid}: Phase 3 reshape already applied",
                          file=sys.stderr)
            except Exception as e:
                pid = os.getpid()
                print(f"[TQ] pid={pid}: Phase 3 reshape patch failed: {e}",
                      file=sys.stderr)

        return mod

    builtins.__import__ = _tq_import_hook
