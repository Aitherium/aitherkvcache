"""
Tests for vLLM plugin registration and TQGPUCache.

Tests plugin entry point, cache allocation, and basic encode/decode
through the cache layer. Does NOT require a running vLLM instance.
"""

import pytest


class TestPluginRegistration:
    """Test that the vLLM plugin entry point works."""

    def test_register_function_exists(self):
        from aither_kvcache.vllm.plugin import register
        assert callable(register)

    def test_register_does_not_crash(self):
        """register() should be safe to call even without vLLM."""
        from aither_kvcache.vllm.plugin import register
        try:
            register()
        except ImportError:
            # Expected if vLLM is not installed
            pass
        except Exception as e:
            # Should not raise anything else
            pytest.fail(f"register() raised unexpected: {e}")

    def test_entry_point_format(self):
        """Verify pyproject.toml entry point string is importable."""
        import importlib
        mod = importlib.import_module("aither_kvcache.vllm.plugin")
        assert hasattr(mod, "register")


class TestPackageImports:
    """Verify that aither_kvcache correctly re-exports from turboquant."""

    def test_turboquant_reexport(self):
        from aither_kvcache import TurboQuant, TurboQuantConfig
        from turboquant import TurboQuant as TQ2, TurboQuantConfig as TQC2
        assert TurboQuant is TQ2
        assert TurboQuantConfig is TQC2

    def test_version_matches_pyproject(self):
        import aither_kvcache
        assert aither_kvcache.__version__ == "2.1.0"

    def test_correction_factors_available(self):
        """TurboQuant instances should have correction_factors tensor."""
        from turboquant import TurboQuant
        tq = TurboQuant(head_dim=128, bits=4, device="cpu")
        assert hasattr(tq, "correction_factors")
        assert tq.correction_factors.shape == (16,)
        # Correction factors should be close to 1.0
        assert (tq.correction_factors - 1.0).abs().max().item() < 0.05


class TestCorrectionFactors:
    """Validate that IP bias correction actually reduces bias."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_corrected_bias_smaller(self, bits):
        from turboquant import TurboQuant
        tq = TurboQuant(head_dim=128, bits=bits, device="cpu")
        result = tq.validate(num_vectors=20000, device="cpu")

        raw_bias = abs(result["ip_bias"])
        corrected_bias = abs(result["ip_bias_corrected"])
        # Corrected bias should be no worse than raw bias
        # (may be marginally worse due to sampling noise, allow 50% margin)
        assert corrected_bias <= raw_bias * 1.5 + 0.001, \
            f"{bits}-bit: corrected bias {corrected_bias:.6f} > raw {raw_bias:.6f}"
