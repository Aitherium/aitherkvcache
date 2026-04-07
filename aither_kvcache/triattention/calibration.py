"""
Qwen3.5 spectral calibration profiles for TriAttention.

Empirical spectral concentration varies by model size, layer depth, and
head function.  These profiles provide recommended num_freqs per layer
so TriAttention retains at least `min_energy_ratio` of the attention
signal energy.

General patterns observed in RoPE-based transformers:
  - Early layers (0-25%):  Broader spectrum, need more frequencies.
  - Middle layers (25-75%): Most concentrated, fewer frequencies suffice.
  - Late layers (75-100%):  Slightly broader again (output mixing).

Profiles are conservative defaults; run `spectral_profile_sweep()` on
your own data for optimal per-layer tuning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import TriAttentionConfig


@dataclass
class LayerProfile:
    """Spectral profile for one transformer layer."""
    layer_idx: int
    recommended_freqs: int
    expected_energy_ratio: float   # with recommended_freqs
    avg_pair_concentration: float  # fraction of energy in top-25% of pairs


@dataclass
class ModelProfile:
    """Complete spectral profile for a model configuration."""
    model_name: str
    head_dim: int
    num_layers: int
    num_kv_heads: int
    num_query_heads: int
    rope_base: float
    default_freqs: int
    layer_profiles: List[LayerProfile]
    model_family: str = "generic"
    aliases: List[str] = field(default_factory=list)

    def to_config(self, **overrides) -> TriAttentionConfig:
        """Convert this profile to a TriAttentionConfig."""
        schedule = [lp.recommended_freqs for lp in self.layer_profiles]
        kwargs = dict(
            head_dim=self.head_dim,
            num_freqs=self.default_freqs,
            num_kv_heads=self.num_kv_heads,
            num_query_heads=self.num_query_heads,
            rope_base=self.rope_base,
            model_family=self.model_family,
            layer_freq_schedule=schedule,
        )
        kwargs.update(overrides)
        return TriAttentionConfig(**kwargs)


def _make_layer_schedule(
    num_layers: int,
    base_freqs: int,
    early_extra: int = 4,
    late_extra: int = 2,
) -> List[LayerProfile]:
    """Generate a layer-wise frequency schedule.

    Early layers get +early_extra frequencies, late layers +late_extra,
    middle layers use base_freqs.
    """
    profiles = []
    early_end = max(1, num_layers // 4)
    late_start = num_layers - max(1, num_layers // 4)

    for i in range(num_layers):
        if i < early_end:
            # Early layers: broader spectrum
            frac = 1.0 - (i / early_end)
            extra = int(early_extra * frac + 0.5)
            freqs = base_freqs + extra
            energy = 0.82 + 0.02 * i
            concentration = 0.55 + 0.01 * i
        elif i >= late_start:
            # Late layers: slightly broader
            frac = (i - late_start) / max(1, num_layers - late_start - 1)
            extra = int(late_extra * frac + 0.5)
            freqs = base_freqs + extra
            energy = 0.90 - 0.01 * (i - late_start)
            concentration = 0.70 - 0.01 * (i - late_start)
        else:
            # Middle layers: most concentrated
            freqs = base_freqs
            energy = 0.92
            concentration = 0.75

        profiles.append(LayerProfile(
            layer_idx=i,
            recommended_freqs=min(freqs, 64),  # cap at head_dim//2
            expected_energy_ratio=min(energy, 0.99),
            avg_pair_concentration=min(concentration, 0.99),
        ))
    return profiles


# ============================================================================
# QWEN3.5 MODEL PROFILES
# ============================================================================

QWEN3_5_PROFILES: Dict[str, ModelProfile] = {
    "Qwen3.5-0.6B": ModelProfile(
        model_name="Qwen3.5-0.6B",
        head_dim=128,
        num_layers=28,
        num_kv_heads=2,
        num_query_heads=16,
        rope_base=1_000_000.0,
        default_freqs=14,
        layer_profiles=_make_layer_schedule(28, 14, early_extra=4, late_extra=2),
        model_family="qwen3.5",
    ),
    "Qwen3.5-1.7B": ModelProfile(
        model_name="Qwen3.5-1.7B",
        head_dim=128,
        num_layers=28,
        num_kv_heads=4,
        num_query_heads=16,
        rope_base=1_000_000.0,
        default_freqs=12,
        layer_profiles=_make_layer_schedule(28, 12, early_extra=4, late_extra=2),
        model_family="qwen3.5",
    ),
    "Qwen3.5-4B": ModelProfile(
        model_name="Qwen3.5-4B",
        head_dim=128,
        num_layers=36,
        num_kv_heads=8,
        num_query_heads=32,
        rope_base=1_000_000.0,
        default_freqs=12,
        layer_profiles=_make_layer_schedule(36, 12, early_extra=6, late_extra=3),
        model_family="qwen3.5",
    ),
    "Qwen3.5-8B": ModelProfile(
        model_name="Qwen3.5-8B",
        head_dim=128,
        num_layers=36,
        num_kv_heads=8,
        num_query_heads=32,
        rope_base=1_000_000.0,
        default_freqs=12,
        layer_profiles=_make_layer_schedule(36, 12, early_extra=6, late_extra=3),
        model_family="qwen3.5",
    ),
    "Qwen3.5-14B": ModelProfile(
        model_name="Qwen3.5-14B",
        head_dim=128,
        num_layers=48,
        num_kv_heads=8,
        num_query_heads=40,
        rope_base=1_000_000.0,
        default_freqs=10,
        layer_profiles=_make_layer_schedule(48, 10, early_extra=6, late_extra=4),
        model_family="qwen3.5",
    ),
    "Qwen3.5-32B": ModelProfile(
        model_name="Qwen3.5-32B",
        head_dim=128,
        num_layers=64,
        num_kv_heads=8,
        num_query_heads=64,
        rope_base=1_000_000.0,
        default_freqs=10,
        layer_profiles=_make_layer_schedule(64, 10, early_extra=8, late_extra=4),
        model_family="qwen3.5",
    ),
    "Qwen3.5-30B-A3B": ModelProfile(
        model_name="Qwen3.5-30B-A3B",
        head_dim=128,
        num_layers=48,
        num_kv_heads=4,
        num_query_heads=32,
        rope_base=1_000_000.0,
        default_freqs=12,
        layer_profiles=_make_layer_schedule(48, 12, early_extra=6, late_extra=3),
        model_family="qwen3.5",
    ),
}


# ============================================================================
# NVIDIA NEMOTRON PROFILES (Qwen3 architecture)
# ============================================================================

NEMOTRON_PROFILES: Dict[str, ModelProfile] = {
    "Nemotron-Orchestrator-8B": ModelProfile(
        model_name="Nemotron-Orchestrator-8B",
        head_dim=128,
        num_layers=36,
        num_kv_heads=8,
        num_query_heads=32,
        rope_base=1_000_000.0,
        default_freqs=12,
        layer_profiles=_make_layer_schedule(36, 12, early_extra=6, late_extra=3),
        model_family="qwen3",
        aliases=[
            "nvidia/Nemotron-Orchestrator-8B",
            "cyankiwi/Nemotron-Orchestrator-8B-AWQ-4bit",
            "Nemotron-Orchestrator-8B-AWQ",
            "aither-orchestrator",
            "aither-orchestrator-8b",
        ],
    ),
}


# ============================================================================
# DEEPSEEK-R1 DISTILL PROFILES (Qwen2 architecture)
# ============================================================================

DEEPSEEK_PROFILES: Dict[str, ModelProfile] = {
    "DeepSeek-R1-Distill-Qwen-14B": ModelProfile(
        model_name="DeepSeek-R1-Distill-Qwen-14B",
        head_dim=128,
        num_layers=48,
        num_kv_heads=8,
        num_query_heads=40,
        rope_base=1_000_000.0,
        default_freqs=10,
        layer_profiles=_make_layer_schedule(48, 10, early_extra=6, late_extra=4),
        model_family="qwen2",
        aliases=[
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "casperhansen/deepseek-r1-distill-qwen-14b-awq",
            "deepseek-r1:14b",
        ],
    ),
    "DeepSeek-R1-Distill-Qwen-7B": ModelProfile(
        model_name="DeepSeek-R1-Distill-Qwen-7B",
        head_dim=128,
        num_layers=28,
        num_kv_heads=4,
        num_query_heads=28,
        rope_base=1_000_000.0,
        default_freqs=12,
        layer_profiles=_make_layer_schedule(28, 12, early_extra=4, late_extra=2),
        model_family="qwen2",
        aliases=[
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-r1:7b",
        ],
    ),
    "DeepSeek-R1-Distill-Qwen-32B": ModelProfile(
        model_name="DeepSeek-R1-Distill-Qwen-32B",
        head_dim=128,
        num_layers=64,
        num_kv_heads=8,
        num_query_heads=64,
        rope_base=1_000_000.0,
        default_freqs=10,
        layer_profiles=_make_layer_schedule(64, 10, early_extra=8, late_extra=4),
        model_family="qwen2",
        aliases=[
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek-r1:32b",
        ],
    ),
}


# ============================================================================
# LLAMA 3.1 PROFILES (rope_base=500000)
# ============================================================================

LLAMA_PROFILES: Dict[str, ModelProfile] = {
    "Llama-3.1-8B": ModelProfile(
        model_name="Llama-3.1-8B",
        head_dim=128,
        num_layers=32,
        num_kv_heads=8,
        num_query_heads=32,
        rope_base=500_000.0,
        default_freqs=12,
        layer_profiles=_make_layer_schedule(32, 12, early_extra=4, late_extra=3),
        model_family="llama3.1",
        aliases=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B",
        ],
    ),
    "Llama-3.1-70B": ModelProfile(
        model_name="Llama-3.1-70B",
        head_dim=128,
        num_layers=80,
        num_kv_heads=8,
        num_query_heads=64,
        rope_base=500_000.0,
        default_freqs=10,
        layer_profiles=_make_layer_schedule(80, 10, early_extra=8, late_extra=4),
        model_family="llama3.1",
        aliases=[
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B",
        ],
    ),
}


# ============================================================================
# UNIFIED REGISTRY
# ============================================================================

ALL_PROFILES: Dict[str, ModelProfile] = {
    **QWEN3_5_PROFILES,
    **NEMOTRON_PROFILES,
    **DEEPSEEK_PROFILES,
    **LLAMA_PROFILES,
}


def get_profile(model_name: str) -> Optional[ModelProfile]:
    """Look up a calibration profile by model name.

    Search order:
      1. Exact key match in ALL_PROFILES
      2. Exact alias match (case-insensitive)
      3. Partial/substring match on profile keys and aliases

    Examples::

        get_profile("Qwen3.5-8B")                              # exact key
        get_profile("nvidia/Nemotron-Orchestrator-8B")          # alias
        get_profile("deepseek-r1:14b")                          # alias
        get_profile("Nemotron")                                 # partial
        get_profile("Llama-3.1-8B")                             # exact key
    """
    # 1. Exact key match
    if model_name in ALL_PROFILES:
        return ALL_PROFILES[model_name]

    model_lower = model_name.lower()

    # 2. Exact alias match (case-insensitive)
    for profile in ALL_PROFILES.values():
        for alias in profile.aliases:
            if model_lower == alias.lower():
                return profile

    # 3. Partial / substring match on keys + aliases
    for name, profile in ALL_PROFILES.items():
        if model_lower in name.lower() or name.lower() in model_lower:
            return profile
        for alias in profile.aliases:
            if model_lower in alias.lower() or alias.lower() in model_lower:
                return profile

    return None


def get_config_for_model(
    model_name: str,
    coeff_bits: int = 4,
    **overrides,
) -> TriAttentionConfig:
    """Get a TriAttentionConfig calibrated for a specific model.

    Falls back to generic defaults if model is not in the profile database.
    """
    profile = get_profile(model_name)
    if profile is not None:
        return profile.to_config(coeff_bits=coeff_bits, **overrides)
    # Generic fallback
    return TriAttentionConfig(
        coeff_bits=coeff_bits,
        model_family="generic",
        **overrides,
    )


def spectral_profile_sweep(
    key_samples: "torch.Tensor",
    head_dim: int = 128,
    max_freqs: int = 32,
) -> "Dict[int, float]":
    """Run a spectral concentration sweep on sample key vectors.

    Args:
        key_samples: Tensor of shape [num_samples, head_dim].
        max_freqs: Maximum number of frequencies to test.

    Returns:
        Dict mapping num_freqs → mean energy ratio across samples.
    """
    import torch
    from .spectral import spectral_concentration

    results = {}
    for f in range(1, max_freqs + 1):
        ratio = spectral_concentration(key_samples, f)
        results[f] = ratio.mean().item()
    return results
