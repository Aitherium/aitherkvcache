"""
GB10 / Blackwell FP8 rescue path for TurboQuant + vLLM.

vLLM 0.19.x can select Cutlass FP8 W8A8 ``scaled_mm`` kernels that were built
for earlier Blackwell/Hopper targets. On DGX Spark / NVIDIA GB10 (sm_121), that
path can fail during engine initialization with ``RuntimeError: Error Internal``.

This module is intentionally part of TurboQuant / AitherKVCache, not a separate
package: it is a KV/inference runtime compatibility layer that protects the
same vLLM deployments TurboQuant extends.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Tuple

_LOG = logging.getLogger("aither.turboquant.gb10")
if not _LOG.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[AitherKVCache] %(message)s"))
    _LOG.addHandler(_handler)
    _LOG.setLevel(os.environ.get("AITHER_TQ_LOG_LEVEL", "INFO").upper())

_STATE = {
    "patched": False,
    "device_name": None,
    "compute_cap": None,
    "patches_applied": [],
    "patches_skipped": [],
    "errors": [],
}

_ORIGINAL_TORCH_SCALED_MM = None


def detect_gpu() -> Optional[Tuple[int, int, str]]:
    """Return ``(major, minor, name)`` for GPU 0, or ``None`` if CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return major, minor, torch.cuda.get_device_name(0)
    except Exception:
        return None


def get_gb10_fp8_status() -> dict:
    """Return a diagnostic snapshot for health checks and startup logs."""
    return dict(_STATE)


def _expand_scale_for_matrix(scale, rows: int, cols: int, role: str):
    """Expand scalar/vector/block FP8 scales to a matrix-compatible shape."""
    import math
    import torch

    scale_fp32 = scale.to(torch.float32) if scale.dtype != torch.float32 else scale
    if scale_fp32.ndim == 0:
        return scale_fp32.view(1, 1)
    if scale_fp32.numel() == 1:
        return scale_fp32.reshape(1, 1)
    if scale_fp32.shape == (rows, cols):
        return scale_fp32
    if scale_fp32.ndim == 1:
        if scale_fp32.numel() == rows:
            return scale_fp32.view(rows, 1)
        if scale_fp32.numel() == cols:
            return scale_fp32.view(1, cols)
    if scale_fp32.ndim == 2:
        scale_rows, scale_cols = scale_fp32.shape
        if scale_rows == rows and scale_cols < cols:
            repeat_cols = math.ceil(cols / scale_cols)
            return scale_fp32.repeat_interleave(repeat_cols, dim=1)[:, :cols]
        if role == "b" and scale_rows < rows and scale_cols < cols:
            repeat_rows = math.ceil(rows / scale_rows)
            repeat_cols = math.ceil(cols / scale_cols)
            return scale_fp32.repeat_interleave(repeat_rows, dim=0).repeat_interleave(repeat_cols, dim=1)[:rows, :cols]
        if scale_rows < rows and scale_cols == cols:
            repeat_rows = math.ceil(rows / scale_rows)
            return scale_fp32.repeat_interleave(repeat_rows, dim=0)[:rows, :]
    return scale_fp32


def _bf16_scaled_mm_fallback(a, b, scale_a, scale_b, out_dtype, bias=None, out=None):
    """Dequantize FP8 inputs with scalar/vector/block scales and run bf16 matmul."""
    import torch

    out_dtype = out_dtype or torch.bfloat16
    a_rows, a_cols = a.shape[-2], a.shape[-1]
    b_rows, b_cols = b.shape[-2], b.shape[-1]
    a_scale = _expand_scale_for_matrix(scale_a, a_rows, a_cols, "a").to(torch.bfloat16)
    b_scale = _expand_scale_for_matrix(scale_b, b_rows, b_cols, "b").to(torch.bfloat16)
    a_bf16 = a.to(torch.bfloat16) * a_scale
    b_bf16 = b.to(torch.bfloat16) * b_scale
    result = torch.matmul(a_bf16, b_bf16).to(out_dtype)
    if bias is not None:
        result = result + bias.to(out_dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _is_torch_compiling() -> bool:
    """Return True while Dynamo/TorchInductor is tracing/compiling a graph."""
    try:
        import torch

        compiler = getattr(torch, "compiler", None)
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling) and is_compiling():
            return True
    except Exception:
        pass
    try:
        import torch._dynamo as dynamo  # type: ignore

        is_compiling = getattr(dynamo, "is_compiling", None)
        return bool(callable(is_compiling) and is_compiling())
    except Exception:
        return False


def _scaled_mm_safe(a, b, scale_a, scale_b, out_dtype, bias=None):
    """
    Cutlass-free FP8 W8A8 scaled matmul.

    Fast path uses PyTorch's cuBLASLt-backed ``torch._scaled_mm``. If that path
    rejects the exact scaling layout, a bf16 dequant/matmul fallback keeps the
    model online and numerically correct instead of crashing the engine.
    """
    import torch

    if _is_torch_compiling():
        return _bf16_scaled_mm_fallback(a, b, scale_a, scale_b, out_dtype, bias=bias)

    try:
        scaled_mm = _ORIGINAL_TORCH_SCALED_MM or torch._scaled_mm
        scale_a_fp32 = scale_a.to(torch.float32) if scale_a.dtype != torch.float32 else scale_a
        scale_b_fp32 = scale_b.to(torch.float32) if scale_b.dtype != torch.float32 else scale_b
        if scale_a_fp32.ndim == 0:
            scale_a_fp32 = scale_a_fp32.view(1)
        if scale_b_fp32.ndim == 0:
            scale_b_fp32 = scale_b_fp32.view(1)
        if scale_a_fp32.ndim == 1 and scale_a_fp32.numel() == a.shape[0]:
            scale_a_fp32 = scale_a_fp32.view(-1, 1).contiguous()
        if scale_b_fp32.ndim == 1 and scale_b_fp32.numel() == b.shape[1]:
            scale_b_fp32 = scale_b_fp32.view(1, -1).contiguous()

        kwargs = {
            "out_dtype": out_dtype,
            "scale_a": scale_a_fp32,
            "scale_b": scale_b_fp32,
        }
        if bias is not None:
            kwargs["bias"] = bias
        return scaled_mm(a, b, **kwargs)
    except Exception as cublas_error:
        first_error = cublas_error

    try:
        return _bf16_scaled_mm_fallback(a, b, scale_a, scale_b, out_dtype, bias=bias)
    except Exception as fallback_error:
        raise RuntimeError(
            "TurboQuant GB10 scaled_mm rescue failed: "
            f"cuBLASLt={first_error!r}; bf16_fallback={fallback_error!r}"
        )


def _torch_scaled_mm_safe(*args, **kwargs):
    """Safe wrapper for TorchInductor/ATen ``_scaled_mm`` calls on GB10."""
    import torch

    original = _ORIGINAL_TORCH_SCALED_MM or torch._scaled_mm
    try:
        return original(*args, **kwargs)
    except Exception as cublas_error:
        if len(args) < 4:
            raise
        a = args[0]
        b = args[1]
        scale_a = kwargs.get("scale_a", args[2] if len(args) > 2 else None)
        scale_b = kwargs.get("scale_b", args[3] if len(args) > 3 else None)
        if scale_a is None or scale_b is None:
            raise
        bias = kwargs.get("bias", args[4] if len(args) > 4 else None)
        out_dtype = kwargs.get("out_dtype") or torch.bfloat16
        out = kwargs.get("out")
        try:
            _LOG.warning("torch._scaled_mm failed on GB10; using bf16 rescue fallback: %s", cublas_error)
            return _bf16_scaled_mm_fallback(a, b, scale_a, scale_b, out_dtype, bias=bias, out=out)
        except Exception as fallback_error:
            raise RuntimeError(
                "TurboQuant GB10 torch._scaled_mm rescue failed: "
                f"cuBLASLt={cublas_error!r}; bf16_fallback={fallback_error!r}"
            )


def install_gb10_fp8_rescue(force: bool = False) -> bool:
    """
    Install the GB10 FP8 rescue patch if the host needs it.

    Environment controls:
    - ``AITHER_TQ_GB10_RESCUE_DISABLE=1``: disable this patch.
    - ``AITHER_TQ_GB10_RESCUE_FORCE=1``: force install on non-sm_12x devices.
    """
    if os.environ.get("AITHER_TQ_GB10_RESCUE_DISABLE", "").lower() in ("1", "true", "yes"):
        _LOG.info("AITHER_TQ_GB10_RESCUE_DISABLE set; skipping FP8 rescue")
        return False
    # If the deployment explicitly enabled the native FP8 kernels (VLLM_DISABLED_KERNELS=""),
    # it is running on an image with working sm_12x cutlass FP8 (e.g. the vLLM 0.22+ nightly).
    # Our cuBLASLt override of cutlass_scaled_mm is then both REDUNDANT and NOT CUDA-graph /
    # stream-capture-safe — it crash-loops hybrid attention+Mamba models (Qwen3.6-27B) on GB10
    # with an async `cudaErrorNotPermitted` in the GDN SSM-state path. Respect the operator's
    # intent (native FP8) and skip the override entirely. Setting VLLM_DISABLED_KERNELS="" is
    # now sufficient; the explicit AITHER_TQ_GB10_RESCUE_DISABLE above remains as a hard override.
    _disabled_kernels = os.environ.get("VLLM_DISABLED_KERNELS")
    if _disabled_kernels is not None and _disabled_kernels.strip() == "":
        _LOG.info(
            "VLLM_DISABLED_KERNELS empty → native FP8 kernels enabled; skipping GB10 "
            "cutlass override (redundant on the nightly and not CUDA-graph-capture-safe)"
        )
        return False
    if _STATE["patched"] and not force:
        return True

    gpu = detect_gpu()
    if gpu is None:
        _LOG.info("No CUDA GPU detected; GB10 FP8 rescue not needed")
        return False

    major, minor, name = gpu
    _STATE["device_name"] = name
    _STATE["compute_cap"] = f"sm_{major}{minor}"

    should_apply = (
        force
        or os.environ.get("AITHER_TQ_GB10_RESCUE_FORCE", "").lower() in ("1", "true", "yes")
        or major >= 12
    )
    if not should_apply:
        _LOG.info("Device %s (sm_%d%d) does not need GB10 FP8 rescue", name, major, minor)
        return False

    _LOG.info("Detected %s (sm_%d%d); installing TurboQuant GB10 FP8 rescue", name, major, minor)

    global _ORIGINAL_TORCH_SCALED_MM

    applied = []
    skipped = []
    errors = []

    try:
        import vllm._custom_ops as custom_ops  # type: ignore

        if hasattr(custom_ops, "cutlass_scaled_mm"):
            def replacement(a, b, scale_a, scale_b, out_dtype, bias=None):
                return _scaled_mm_safe(a, b, scale_a, scale_b, out_dtype, bias)

            custom_ops.cutlass_scaled_mm = replacement
            applied.append("vllm._custom_ops.cutlass_scaled_mm")
        else:
            skipped.append("vllm._custom_ops.cutlass_scaled_mm (symbol absent)")
    except ImportError:
        skipped.append("vllm._custom_ops (vllm not installed)")
    except Exception as error:
        errors.append(f"vllm._custom_ops.cutlass_scaled_mm: {error!r}")
        _LOG.warning("Could not patch vllm._custom_ops.cutlass_scaled_mm: %s", error)

    try:
        import torch

        if hasattr(torch, "_scaled_mm"):
            current_scaled_mm = torch._scaled_mm
            if _ORIGINAL_TORCH_SCALED_MM is None:
                _ORIGINAL_TORCH_SCALED_MM = current_scaled_mm

        try:
            from torch._inductor.select_algorithm import extern_kernels  # type: ignore

            if hasattr(extern_kernels, "_scaled_mm"):
                extern_kernels._scaled_mm = _torch_scaled_mm_safe
                applied.append("torch._inductor.extern_kernels._scaled_mm")
        except Exception as error:
            skipped.append(f"torch._inductor.extern_kernels._scaled_mm ({error})")

        if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "cutlass_scaled_mm"):
            library = torch.library.Library("_C", "IMPL")

            def cuda_impl(a, b, scale_a, scale_b, out_dtype, bias=None):
                return _scaled_mm_safe(a, b, scale_a, scale_b, out_dtype, bias)

            library.impl("cutlass_scaled_mm", cuda_impl, "CUDA")
            applied.append("torch.ops._C.cutlass_scaled_mm[CUDA]")
        else:
            skipped.append("torch.ops._C.cutlass_scaled_mm (op not registered yet)")
    except RuntimeError as error:
        skipped.append(f"torch.ops._C.cutlass_scaled_mm[CUDA] ({error})")
    except Exception as error:
        errors.append(f"torch.ops._C.cutlass_scaled_mm: {error!r}")
        _LOG.warning("Could not patch torch.ops._C.cutlass_scaled_mm: %s", error)

    _STATE["patches_applied"] = applied
    _STATE["patches_skipped"] = skipped
    _STATE["errors"] = errors
    _STATE["patched"] = bool(applied) and not errors

    if _STATE["patched"]:
        _LOG.info("TurboQuant GB10 FP8 rescue INSTALLED (%d patches)", len(applied))
    else:
        _LOG.warning("TurboQuant GB10 FP8 rescue incomplete: applied=%s errors=%s", applied, errors)
    return _STATE["patched"]