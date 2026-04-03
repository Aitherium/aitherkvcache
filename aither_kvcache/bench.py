"""
TurboQuant Benchmark — Validate correctness and measure performance.

Run directly:
    python -m lib.gpu.turboquant.bench

Or from repo root:
    python AitherOS/lib/gpu/turboquant/bench.py
"""

import sys
import os
import time

# Allow running from repo root
if os.path.exists("AitherOS"):
    sys.path.insert(0, "AitherOS")

import torch
from lib.gpu.turboquant import TurboQuant, TurboQuantConfig


def banner(text: str):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def run_validation():
    banner("CORRECTNESS VALIDATION")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"VRAM: {vram:.1f} GB")
    print()

    for bits in [4, 3, 2]:
        tq = TurboQuant(head_dim=128, bits=bits, device=device)
        result = tq.validate(num_vectors=50000, device=device)

        status = "PASS" if result["mse"] <= result["mse_theory_upper"] * 1.1 else "FAIL"
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


def run_throughput():
    banner("THROUGHPUT BENCHMARK")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_vectors = 32768 if device == "cuda" else 4096

    for bits in [4, 3, 2]:
        tq = TurboQuant(head_dim=128, bits=bits, device=device)
        result = tq.benchmark(num_vectors=n_vectors, device=device)
        print(f"[{bits}-bit] {n_vectors} vectors, {device}")
        print(f"  Encode: {result['encode_us']:.0f} us "
              f"({result['encode_throughput_mvec_s']:.2f} Mvec/s)")
        print(f"  Decode: {result['decode_us']:.0f} us "
              f"({result['decode_throughput_mvec_s']:.2f} Mvec/s)")
        print()


def run_memory_report():
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


def run_context_impact():
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


if __name__ == "__main__":
    print("TurboQuant v0.1.0 — KV Cache Quantization Benchmark")
    print("Paper: arXiv:2504.19874 (Zandieh et al., 2025)")

    run_validation()
    run_throughput()
    run_memory_report()
    run_context_impact()

    banner("DONE")
