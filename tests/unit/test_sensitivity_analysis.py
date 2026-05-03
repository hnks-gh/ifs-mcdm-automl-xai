"""
tests/unit/test_sensitivity_analysis.py
========================================
Unit tests for Monte Carlo sensitivity analysis module.

Test coverage:
- Dirichlet weight sampling (distribution properties, simplex constraint)
- Kendall tau-b (weighted variant) calculation
- Weight perturbation pipeline
- Sensitivity result aggregation and statistics
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from src.core.exceptions import FrameworkError
from src.core.schema import WeightVector, RankingMethod
from src.mcdm.analysis.sensitivity_analysis import (
    sample_dirichlet_weights,
    kendall_tau_b_weighted,
    SensitivityResult,
)


class TestDirichletSampling:
    """Tests for Dirichlet weight sampling."""

    def test_dirichlet_samples_are_valid_weights(self):
        """All samples should be valid probability distributions."""
        base_weights = np.array([0.4, 0.3, 0.3])
        samples = sample_dirichlet_weights(base_weights, n_samples=100, random_state=42)

        assert samples.shape == (100, 3)
        
        # Check sum to 1
        sums = np.sum(samples, axis=1)
        assert np.allclose(sums, 1.0, atol=1e-9)
        
        # Check non-negative
        assert np.all(samples >= 0.0)

    def test_dirichlet_base_not_normalized(self):
        """Should error if base weights don't sum to 1."""
        base_weights = np.array([0.5, 0.3, 0.1])  # Sums to 0.9
        with pytest.raises(ValueError):
            sample_dirichlet_weights(base_weights, n_samples=100)

    def test_dirichlet_base_with_negative(self):
        """Should error if base weights contain negative values."""
        base_weights = np.array([-0.1, 0.6, 0.5])
        with pytest.raises(ValueError):
            sample_dirichlet_weights(base_weights, n_samples=100)

    def test_dirichlet_high_alpha_low_variance(self):
        """High α_scale should produce low variance samples."""
        base_weights = np.array([0.5, 0.5])
        
        # High α_scale: low variance
        samples_high = sample_dirichlet_weights(
            base_weights, alpha_scale=100.0, n_samples=1000, random_state=42
        )
        std_high = np.std(samples_high, axis=0)
        
        # Low α_scale: high variance
        samples_low = sample_dirichlet_weights(
            base_weights, alpha_scale=1.0, n_samples=1000, random_state=42
        )
        std_low = np.std(samples_low, axis=0)
        
        # High α should have lower variance
        assert np.all(std_high < std_low)

    def test_dirichlet_mean_close_to_base(self):
        """Mean of samples should be close to base distribution."""
        base_weights = np.array([0.4, 0.3, 0.3])
        samples = sample_dirichlet_weights(
            base_weights, alpha_scale=50.0, n_samples=10000, random_state=42
        )
        sample_mean = np.mean(samples, axis=0)
        
        assert np.allclose(sample_mean, base_weights, atol=0.02)

    def test_dirichlet_reproducibility(self):
        """Same random_state should produce identical samples."""
        base_weights = np.array([0.5, 0.5])
        
        samples1 = sample_dirichlet_weights(base_weights, n_samples=100, random_state=42)
        samples2 = sample_dirichlet_weights(base_weights, n_samples=100, random_state=42)
        
        assert np.allclose(samples1, samples2)

    def test_dirichlet_different_seeds(self):
        """Different random_state should produce different samples."""
        base_weights = np.array([0.5, 0.5])
        
        samples1 = sample_dirichlet_weights(base_weights, n_samples=100, random_state=42)
        samples2 = sample_dirichlet_weights(base_weights, n_samples=100, random_state=43)
        
        # Should not be identical
        assert not np.allclose(samples1, samples2)

    def test_dirichlet_uniform_base(self):
        """Dirichlet of uniform base should still be valid."""
        base_weights = np.array([0.25, 0.25, 0.25, 0.25])
        samples = sample_dirichlet_weights(base_weights, n_samples=100)
        
        sums = np.sum(samples, axis=1)
        assert np.allclose(sums, 1.0, atol=1e-9)


class TestKendallTauB:
    """Tests for weighted Kendall's tau-b computation."""

    def test_kendall_identical_rankings(self):
        """τ_b of identical rankings should be +1."""
        rank1 = [1, 2, 3, 4, 5]
        rank2 = [1, 2, 3, 4, 5]
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        tau_b = kendall_tau_b_weighted(rank1, rank2, weights)
        assert math.isclose(tau_b, 1.0, abs_tol=1e-9)

    def test_kendall_reverse_rankings(self):
        """τ_b of reverse rankings should be -1."""
        rank1 = [1, 2, 3, 4, 5]
        rank2 = [5, 4, 3, 2, 1]
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        tau_b = kendall_tau_b_weighted(rank1, rank2, weights)
        assert math.isclose(tau_b, -1.0, abs_tol=1e-9)

    def test_kendall_partial_agreement(self):
        """τ_b with partial agreement should be in (-1, 1)."""
        rank1 = [1, 2, 3, 4, 5]
        rank2 = [2, 1, 3, 4, 5]  # Swap 1 and 2
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        tau_b = kendall_tau_b_weighted(rank1, rank2, weights)
        assert -1.0 <= tau_b <= 1.0
        assert not math.isclose(tau_b, 1.0)  # Not perfect agreement

    def test_kendall_symmetry(self):
        """τ_b should be symmetric: τ_b(r1, r2) = τ_b(r2, r1)."""
        rank1 = [1, 3, 2, 5, 4]
        rank2 = [2, 1, 3, 4, 5]
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        tau_b_12 = kendall_tau_b_weighted(rank1, rank2, weights)
        tau_b_21 = kendall_tau_b_weighted(rank2, rank1, weights)
        
        assert math.isclose(tau_b_12, tau_b_21, abs_tol=1e-9)

    def test_kendall_weighted_effect(self):
        """Weights should affect disagreement significance."""
        rank1 = [1, 2, 3]
        rank2 = [2, 1, 3]  # Swap first two items
        
        # Equal weights
        tau_b_equal = kendall_tau_b_weighted(rank1, rank2, np.array([1.0/3, 1.0/3, 1.0/3]))
        
        # Weights favoring first item
        tau_b_weighted = kendall_tau_b_weighted(rank1, rank2, np.array([0.8, 0.1, 0.1]))
        
        # Both should have positive tau_b (2 concordant, 1 discordant pairs)
        # tau_b = (C - D) / (C + D) = (2 - 1) / 3 = 1/3 ≈ 0.33
        assert tau_b_equal > 0.0
        assert tau_b_weighted > 0.0
        # With higher weight on disagreeing item (0.8 on first item), 
        # the pair (0,1) which is discordant becomes more significant
        # This should make tau_b lower when we weight the disagreement more heavily
        assert tau_b_weighted < tau_b_equal

    def test_kendall_mismatched_lengths(self):
        """Should error on length mismatch."""
        rank1 = [1, 2, 3]
        rank2 = [1, 2]
        weights = np.array([0.33, 0.33, 0.34])
        
        with pytest.raises(ValueError):
            kendall_tau_b_weighted(rank1, rank2, weights)

    def test_kendall_invalid_rank_values(self):
        """Should error if ranks are not permutations of 1..n."""
        rank1 = [1, 2, 3]
        rank2 = [1, 2, 2]  # Duplicate rank
        weights = np.array([0.33, 0.33, 0.34])
        
        with pytest.raises(ValueError):
            kendall_tau_b_weighted(rank1, rank2, weights)

    def test_kendall_gaps_in_ranks(self):
        """Should error if ranks have gaps."""
        rank1 = [1, 3, 5]  # Gap
        rank2 = [1, 2, 3]
        weights = np.array([0.33, 0.33, 0.34])
        
        with pytest.raises(ValueError):
            kendall_tau_b_weighted(rank1, rank2, weights)

    def test_kendall_scaling_weights(self):
        """Relative weight scaling should not change τ_b (normalized)."""
        rank1 = [1, 2, 3, 4]
        rank2 = [2, 1, 3, 4]
        
        weights1 = np.array([1.0, 1.0, 1.0, 1.0])
        weights2 = np.array([2.0, 2.0, 2.0, 2.0])
        
        tau_b_1 = kendall_tau_b_weighted(rank1, rank2, weights1)
        tau_b_2 = kendall_tau_b_weighted(rank1, rank2, weights2)
        
        # Should be equivalent since weights are normalized
        assert math.isclose(tau_b_1, tau_b_2, abs_tol=1e-9)

    def test_kendall_ties_not_in_ranks(self):
        """Standard ranks should not have ties."""
        rank1 = [1, 1, 2]  # Tie
        rank2 = [1, 2, 3]
        weights = np.array([0.33, 0.33, 0.34])
        
        with pytest.raises(ValueError):
            kendall_tau_b_weighted(rank1, rank2, weights)


class TestSensitivityResult:
    """Tests for SensitivityResult dataclass."""

    def test_result_to_dict(self):
        """Test serialization to dictionary."""
        result = SensitivityResult(
            baseline_ranks={
                RankingMethod.IF_WASPAS: [1, 2, 3, 4, 5],
                RankingMethod.IF_TOPSIS: [1, 2, 3, 4, 5],
            },
            n_simulations=1000,
            perturbation_scale=10.0,
            random_state=42,
            kendall_tau_b_distributions={
                RankingMethod.IF_WASPAS: [0.95, 0.92, 0.98],
                RankingMethod.IF_TOPSIS: [0.90, 0.88, 0.93],
            },
            summary_stats={
                RankingMethod.IF_WASPAS: {
                    "mean": 0.95,
                    "std": 0.02,
                    "ci_lower": 0.91,
                    "ci_upper": 0.99,
                },
                RankingMethod.IF_TOPSIS: {
                    "mean": 0.90,
                    "std": 0.02,
                    "ci_lower": 0.86,
                    "ci_upper": 0.94,
                },
            },
        )

        result_dict = result.to_dict()
        assert result_dict["n_simulations"] == 1000
        assert result_dict["random_state"] == 42
        assert "if_waspas" in result_dict["baseline_ranks"]

    def test_result_to_dataframe(self):
        """Test conversion to DataFrames."""
        result = SensitivityResult(
            baseline_ranks={
                RankingMethod.IF_WASPAS: [1, 2, 3],
                RankingMethod.IF_TOPSIS: [1, 2, 3],
            },
            n_simulations=100,
            perturbation_scale=10.0,
            random_state=42,
            kendall_tau_b_distributions={
                RankingMethod.IF_WASPAS: [0.95, 0.92, 0.98],
                RankingMethod.IF_TOPSIS: [0.90, 0.88, 0.93],
            },
            summary_stats={
                RankingMethod.IF_WASPAS: {
                    "mean": 0.95,
                    "std": 0.02,
                    "ci_lower": 0.91,
                    "ci_upper": 0.99,
                },
                RankingMethod.IF_TOPSIS: {
                    "mean": 0.90,
                    "std": 0.02,
                    "ci_lower": 0.86,
                    "ci_upper": 0.94,
                },
            },
        )

        tau_b_df, stats_df = result.to_dataframe()
        
        assert tau_b_df.shape == (3, 2)  # 3 simulations, 2 methods
        assert stats_df.shape == (2, 5)  # 2 methods, 5 stat columns


class TestWeightPerturbationPipeline:
    """Integration tests for weight perturbation pipeline."""

    def test_perturbation_produces_valid_weights(self):
        """Perturbed weights should always be valid (sum to 1, non-negative)."""
        base_weights = np.array([0.3, 0.4, 0.3])
        
        for alpha_scale in [1.0, 5.0, 10.0, 50.0]:
            samples = sample_dirichlet_weights(base_weights, alpha_scale, n_samples=100)
            
            # All rows sum to 1
            sums = np.sum(samples, axis=1)
            assert np.allclose(sums, 1.0, atol=1e-9)
            
            # All entries non-negative
            assert np.all(samples >= 0.0)

    def test_kendall_with_random_perturbations(self):
        """Kendall tau-b should be computable for any random perturbation."""
        base_rank = [1, 2, 3, 4, 5]
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        base_weights_dist = np.array([0.5, 0.5])
        perturbed = sample_dirichlet_weights(base_weights_dist, n_samples=100, random_state=42)
        
        # For each perturbation, create a "perturbed rank" (shuffle based on perturbation)
        rng = np.random.RandomState(42)
        for i in range(min(10, len(perturbed))):
            perturbed_rank = base_rank.copy()
            # Randomly reorder based on perturbation weights
            if rng.uniform() < 0.3:
                perturbed_rank[0], perturbed_rank[1] = perturbed_rank[1], perturbed_rank[0]
            
            tau_b = kendall_tau_b_weighted(base_rank, perturbed_rank, weights)
            assert -1.0 <= tau_b <= 1.0


class TestSensitivityStatistics:
    """Tests for statistical summary computations in sensitivity analysis."""

    def test_percentile_ci_computation(self):
        """Test confidence interval computation from tau-b distribution."""
        # Create synthetic tau-b distribution (normal-ish)
        tau_b_vals = np.random.normal(loc=0.95, scale=0.05, size=1000)
        tau_b_vals = np.clip(tau_b_vals, -1.0, 1.0)
        
        mean = np.mean(tau_b_vals)
        std = np.std(tau_b_vals, ddof=1)
        ci_lower = np.percentile(tau_b_vals, 2.5)
        ci_upper = np.percentile(tau_b_vals, 97.5)
        
        # CI should be roughly ±1.96*std around mean for 95% CI
        # For normal dist: roughly mean ± 1.96*std
        assert ci_lower < mean
        assert ci_upper > mean
        assert ci_upper - ci_lower > std  # Width > 1 std


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
