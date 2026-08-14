"""
TurboQuant Benchmark — Validate correctness and measure performance.

Run directly:
    python -m lib.gpu.turboquant.bench
    python -m lib.gpu.turboquant.bench --strict    # harness mode

Or from repo root:
    python -m turboquant.bench

MACHINE-READABLE OUTPUT
    Every measurement is also emitted as a single stable line:

        METRIC <name>=<float>

    That line is the contract AitherAutoResearch's ratchet harness parses
    (`metric_regex`). The headline the ratchet maximises is
    `turboquant_decode_mvec_s` (4-bit decode throughput).

--strict IS THE HARNESS MODE
    run_validation() has always COMPUTED a PASS/FAIL against the paper's MSE
    bound and then thrown it away — it printed the word and exited 0. A
    quantizer that got faster by quantising worse would therefore have scored
    as an improvement and been kept. Under --strict a FAIL exits 1, so the
    ratchet reverts it, and an absent CUDA device fails rather than quietly
    ratcheting on CPU numbers.
"""

import argparse
import os
import sys

# Allow running from repo root
if os.path.exists("AitherOS"):
    sys.path.insert(0, "AitherOS")

import torch  # noqa: E402
from lib.gpu.turboquant import TurboQuant, TurboQuantConfig  # noqa: E402,F401

# Collected as (name, value) so main() can emit one stable METRIC block.
_METRICS: list[tuple[str, float]] = []


def _record(name: str, value: float) -> None:
    """Record a metric for the machine-readable block main() prints."""
    _METRICS.append((name, float(value)))


def banner(text: str):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def run_validation() -> bool:
    """Validate quantization error against the paper's bound.

    Returns False if any bit-width exceeds its theoretical upper bound.
    """
    banner("CORRECTNESS VALIDATION")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        # `total_memory`, not `total_mem` — the old spelling raised
        # AttributeError on every CUDA box, i.e. this validation could only
        # ever have completed on CPU.
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"VRAM: {vram:.1f} GB")
    print()

    ok = True
    for bits in [4, 3, 2]:
        tq = TurboQuant(head_dim=128, bits=bits, device=device)
        result = tq.validate(num_vectors=50000, device=device)

        passed = result["mse"] <= result["mse_theory_upper"] * 1.1
        ok = passed and ok
        status = "PASS" if passed else "FAIL"
        _record(f"tq{bits}_mse", result["mse"])
        _record(f"tq{bits}_mse_ratio_to_lower", result["mse_ratio_to_lower"])
        print(f"[{bits}-bit] {status}")
        print(f"  MSE:          {result['mse']:.6f}")
        print(f"  Theory range: [{result['mse_theory_lower']:.6f}, "
              f"{result['mse_theory_upper']:.6f}]")
        print(f"  Ratio to LB:  {result['mse_ratio_to_lower']:.2f}x "
              f"(paper claims <= 2.7x)")
        print(f"  IP bias:      {result['ip_bias']:.6f}")
        print(f"  IP MSE:       {result['ip_mse']:.6f}")
        print(f"  Compression:  {result['compression_vs_fp16']:.1f}x vs FP16, "
              f"{result['compression_vs_fp8']:.1f}x vs FP8")
        print(f"  Engine:       {'Triton' if result['triton_active'] else 'PyTorch'}")
        print()

    return ok


def run_throughput() -> bool:
    banner("THROUGHPUT BENCHMARK")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_vectors = 32768 if device == "cuda" else 4096

    for bits in [4, 3, 2]:
        tq = TurboQuant(head_dim=128, bits=bits, device=device)
        result = tq.benchmark(num_vectors=n_vectors, device=device)
        _record(f"tq{bits}_encode_us", result["encode_us"])
        _record(f"tq{bits}_decode_us", result["decode_us"])
        _record(f"tq{bits}_decode_mvec_s", result["decode_throughput_mvec_s"])
        if bits == 4:
            # The headline the ratchet maximises: 4-bit is the deployed config.
            _record("turboquant_decode_mvec_s", result["decode_throughput_mvec_s"])
        print(f"[{bits}-bit] {n_vectors} vectors, {device}")
        print(f"  Encode: {result['encode_us']:.0f} us "
              f"({result['encode_throughput_mvec_s']:.2f} Mvec/s)")
        print(f"  Decode: {result['decode_us']:.0f} us "
              f"({result['decode_throughput_mvec_s']:.2f} Mvec/s)")
        print()
    return True


def run_memory_report() -> bool:
    banner("KV CACHE MEMORY REPORT")

    configs = [
        ("Nemotron-8B (orchestrator)", 32, 8, 128),
        ("DeepSeek-R1-14B (reasoning)", 40, 8, 128),
        ("Llama-3.1-70B (supernode)", 80, 8, 128),
    ]

    for model_name, num_layers, num_kv_heads, head_dim in configs:
        print(f"\n  {model_name} (L={num_layers}, KVH={num_kv_heads}, d={head_dim})")
        print(f"  {'Seq Len':>10} {'FP16':>10} {'FP8':>10} "
              f"{'TQ4':>10} {'TQ3':>10} {'TQ2':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for seq_len in [8192, 16384, 32768, 65536, 131072]:
            reports = {}
            for bits in [4, 3, 2]:
                tq = TurboQuant(head_dim=head_dim, bits=bits, device="cpu")
                reports[bits] = tq.memory_report(seq_len, num_layers, num_kv_heads)

            fp16_mb = reports[4]["fp16_mb"]
            fp8_mb = reports[4]["fp8_mb"]
            tq4_mb = reports[4]["tq4_mb"]
            tq3_mb = reports[3]["tq3_mb"]
            tq2_mb = reports[2]["tq2_mb"]

            print(f"  {seq_len:>10,} {fp16_mb:>8.1f}MB {fp8_mb:>8.1f}MB "
                  f"{tq4_mb:>8.1f}MB {tq3_mb:>8.1f}MB {tq2_mb:>8.1f}MB")
    return True


def run_context_impact() -> bool:
    banner("CONTEXT WINDOW IMPACT (RTX 5090, 32GB)")

    # Rough model: available_kv_vram = total_vram * util - model_weights
    # KV per token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_value

    scenarios = [
        {
            "name": "Nemotron-8B Orchestrator (util=0.40)",
            "vram_gb": 32, "util": 0.40, "weights_gb": 4.5,
            "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        },
        {
            "name": "DeepSeek-R1-14B Reasoning (util=0.90, enforce-eager)",
            "vram_gb": 32, "util": 0.90, "weights_gb": 9.4,
            "num_layers": 40, "num_kv_heads": 8, "head_dim": 128,
        },
    ]

    for s in scenarios:
        kv_vram = (s["vram_gb"] * s["util"] - s["weights_gb"]) * 1024  # MB
        vecs_per_token = 2 * s["num_layers"] * s["num_kv_heads"]

        print(f"\n  {s['name']}")
        print(f"  Available KV VRAM: {kv_vram:.0f} MB")
        print()

        for label, bytes_per_val in [("FP16", 256), ("FP8 (current)", 128),
                                      ("TQ4", 68), ("TQ3", 52), ("TQ2", 36)]:
            bytes_per_token = vecs_per_token * bytes_per_val
            max_tokens = int(kv_vram * 1024 * 1024 / bytes_per_token)
            print(f"    {label:>16}: {max_tokens:>8,} tokens "
                  f"({max_tokens/1024:.0f}K)")
    return True


def main(argv: list[str] | None = None) -> int:
    """Run the full bench. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="TurboQuant benchmark")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Harness mode: exit non-zero when a bit-width exceeds its MSE "
            "bound or when CUDA is unavailable."
        ),
    )
    args = parser.parse_args(argv)

    print("TurboQuant v0.1.0 — KV Cache Quantization Benchmark")
    print("Paper: arXiv:2504.19874 (Zandieh et al., 2025)")

    ok = True
    for stage in (run_validation, run_throughput, run_memory_report, run_context_impact):
        try:
            ok = stage() and ok
        except Exception as exc:  # noqa: BLE001 - reported, then failed on
            print(f"\n  {stage.__name__}: ERROR {type(exc).__name__}: {exc}")
            ok = False

    print()
    for name, value in _METRICS:
        print(f"METRIC {name}={value:.6f}")

    if not args.strict:
        banner("DONE")
        return 0

    if not torch.cuda.is_available():
        print(
            "STRICT FAIL: no CUDA device - CPU throughput is not a substitute "
            "measurement for the quantizer under test."
        )
        return 1
    if "turboquant_decode_mvec_s" not in {name for name, _ in _METRICS}:
        print("STRICT FAIL: headline metric turboquant_decode_mvec_s not measured.")
        return 1
    if not ok:
        print("STRICT FAIL: a correctness check failed.")
        return 1

    banner("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
