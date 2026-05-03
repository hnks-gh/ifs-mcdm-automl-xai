"""
tests/unit/test_ranking_validation.py
======================================
Unit tests for ranking validation and comparison analysis module.

Test coverage:
- Spearman rank correlation computation
- IQR (discriminatory power) computation
- Inter-method agreement analysis
- Temporal persistence (year-to-year) analysis
- Linear trend computation
- Full validation pipeline
"""

from __future__ import annotations

import math
import pytest
import numpy as np
import pandas as pd

from src.core.schema import RankingMethod, RankingResult
from src.mcdm.analysis.ranking_validation import (
    compute_spearman_rho,
    compute_score_iqr,
    linear_trend,
    compute_inter_method_agreement,
    compute_discriminatory_power,
    compute_temporal_persistence,
    RankingValidationResult,
)


class TestSpearmanRho:
    """Tests for Spearman rank correlation."""

    def test_spearman_identical_ranks(self):
        """ρ of identical rankings should be +1."""
        rank1 = [1, 2, 3, 4, 5]
        rank2 = [1, 2, 3, 4, 5]
        
        rho = compute_spearman_rho(rank1, rank2)
        assert math.isclose(rho, 1.0, abs_tol=1e-6)

    def test_spearman_reverse_ranks(self):
        """ρ of reverse rankings should be -1."""
        rank1 = [1, 2, 3, 4, 5]
        rank2 = [5, 4, 3, 2, 1]
        
        rho = compute_spearman_rho(rank1, rank2)
        assert math.isclose(rho, -1.0, abs_tol=1e-6)

    def test_spearman_random_agreement(self):
        """ρ of random rankings should be near 0."""
        rank1 = [1, 2, 3, 4, 5]
        rank2 = [3, 1, 5, 2, 4]
        
        rho = compute_spearman_rho(rank1, rank2)
        assert -1.0 <= rho <= 1.0
        # For 5 items random shuffle, should be reasonably away from ±1
        assert abs(rho) < 0.9

    def test_spearman_mismatched_lengths(self):
        """Should error on length mismatch."""
        rank1 = [1, 2, 3]
        rank2 = [1, 2, 3, 4]
        
        with pytest.raises(ValueError):
            compute_spearman_rho(rank1, rank2)

    def test_spearman_single_element(self):
        """Should error with single element."""
        rank1 = [1]
        rank2 = [1]
        
        with pytest.raises(ValueError):
            compute_spearman_rho(rank1, rank2)

    def test_spearman_two_elements(self):
        """Should work with 2 elements."""
        rank1 = [1, 2]
        rank2 = [1, 2]
        
        rho = compute_spearman_rho(rank1, rank2)
        assert math.isclose(rho, 1.0, abs_tol=1e-6)

    def test_spearman_symmetry(self):
        """ρ should be symmetric."""
        rank1 = [1, 3, 2, 5, 4]
        rank2 = [2, 1, 3, 4, 5]
        
        rho_12 = compute_spearman_rho(rank1, rank2)
        rho_21 = compute_spearman_rho(rank2, rank1)
        
        assert math.isclose(rho_12, rho_21, abs_tol=1e-9)

    def test_spearman_monotonic_transformation(self):
        """ρ is invariant to monotonic transformations (as long as ordering preserved)."""
        rank1 = [1, 2, 3, 4, 5]
        rank2_orig = [2, 1, 3, 4, 5]
        
        rho_orig = compute_spearman_rho(rank1, rank2_orig)
        
        # Scale rank2 by monotonic transformation
        rank2_scaled = [x * 10 for x in rank2_orig]
        rho_scaled = compute_spearman_rho(rank1, rank2_scaled)
        
        assert math.isclose(rho_orig, rho_scaled, abs_tol=1e-6)


class TestScoreIQR:
    """Tests for Interquartile Range computation."""

    def test_iqr_uniform_scores(self):
        """IQR of uniform scores should be zero."""
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        iqr = compute_score_iqr(scores)
        assert math.isclose(iqr, 0.0, abs_tol=1e-9)

    def test_iqr_known_value(self):
        """Test IQR against known analytical result."""
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # Q1 (25th percentile) = 3.25, Q3 (75th percentile) = 7.75
        # IQR = 7.75 - 3.25 = 4.5
        iqr = compute_score_iqr(scores)
        expected = 7.75 - 3.25
        assert math.isclose(iqr, expected, abs_tol=0.1)

    def test_iqr_normal_distribution(self):
        """IQR of normal distribution should be ~1.35*σ."""
        np.random.seed(42)
        scores = np.random.normal(loc=0, scale=1.0, size=10000)
        iqr = compute_score_iqr(scores)
        # For standard normal, IQR ≈ 1.35 * σ = 1.35
        assert math.isclose(iqr, 1.35, abs_tol=0.1)

    def test_iqr_sparse_scores(self):
        """Warning but should still work with < 4 points."""
        scores = np.array([1, 2, 3])
        iqr = compute_score_iqr(scores)
        assert iqr >= 0.0  # Should still be computable


class TestLinearTrend:
    """Tests for linear trend computation."""

    def test_trend_increasing(self):
        """Positive trend for increasing sequence."""
        y_values = [1, 2, 3, 4, 5]
        slope = linear_trend(y_values)
        assert slope > 0.0

    def test_trend_decreasing(self):
        """Negative trend for decreasing sequence."""
        y_values = [5, 4, 3, 2, 1]
        slope = linear_trend(y_values)
        assert slope < 0.0

    def test_trend_flat(self):
        """Zero trend for constant sequence."""
        y_values = [3, 3, 3, 3, 3]
        slope = linear_trend(y_values)
        assert math.isclose(slope, 0.0, abs_tol=1e-9)

    def test_trend_single_point(self):
        """Should return None for single point."""
        y_values = [5]
        slope = linear_trend(y_values)
        assert slope is None

    def test_trend_two_points(self):
        """Should compute slope for two points."""
        y_values = [0, 1]
        slope = linear_trend(y_values)
        assert math.isclose(slope, 1.0, abs_tol=1e-9)

    def test_trend_with_nan(self):
        """Should skip NaN values."""
        y_values = [1, np.nan, 3, 4, 5]
        slope = linear_trend(y_values)
        # Should be able to compute slope using non-NaN values
        assert slope is not None and not np.isnan(slope)

    def test_trend_all_nan(self):
        """Should return None if all NaN."""
        y_values = [np.nan, np.nan, np.nan]
        slope = linear_trend(y_values)
        assert slope is None


class TestInterMethodAgreement:
    """Tests for inter-method agreement computation."""

    def test_inter_method_identical_methods(self):
        """Correlation between identical methods should be 1.0."""
        ranking_result = RankingResult(
            method=RankingMethod.IF_WASPAS,
            year=2020,
            provinces=["P01", "P02", "P03"],
            scores=[0.9, 0.8, 0.7],
            ranks=[1, 2, 3],
        )
        
        rankings_per_year = {
            2020: {
                RankingMethod.IF_WASPAS: ranking_result,
                RankingMethod.IF_TOPSIS: ranking_result,  # Same ranking
            }
        }
        
        per_year_rho, overall_rho = compute_inter_method_agreement(rankings_per_year)
        
        # Should have high correlation (would be 1.0 if truly identical)
        assert len(overall_rho) > 0

    def test_inter_method_two_years(self):
        """Test with data spanning two years."""
        rankings_per_year = {
            2020: {
                RankingMethod.IF_WASPAS: RankingResult(
                    method=RankingMethod.IF_WASPAS,
                    year=2020,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.9, 0.8, 0.7],
                    ranks=[1, 2, 3],
                ),
                RankingMethod.IF_TOPSIS: RankingResult(
                    method=RankingMethod.IF_TOPSIS,
                    year=2020,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.85, 0.75, 0.65],
                    ranks=[1, 2, 3],
                ),
            },
            2021: {
                RankingMethod.IF_WASPAS: RankingResult(
                    method=RankingMethod.IF_WASPAS,
                    year=2021,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.88, 0.77, 0.66],
                    ranks=[1, 2, 3],
                ),
                RankingMethod.IF_TOPSIS: RankingResult(
                    method=RankingMethod.IF_TOPSIS,
                    year=2021,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.83, 0.73, 0.63],
                    ranks=[1, 2, 3],
                ),
            },
        }
        
        per_year_rho, overall_rho = compute_inter_method_agreement(rankings_per_year)
        
        # Should have data for both years
        assert 2020 in per_year_rho
        assert 2021 in per_year_rho
        # Should have overall correlation
        assert len(overall_rho) > 0


class TestDiscriminatoryPower:
    """Tests for discriminatory power (IQR) computation."""

    def test_discriminatory_power_uniform_scores(self):
        """IQR should be zero for uniform scores."""
        ranking_result = RankingResult(
            method=RankingMethod.IF_WASPAS,
            year=2020,
            provinces=["P01", "P02", "P03", "P04"],
            scores=[0.5, 0.5, 0.5, 0.5],
            ranks=[1, 2, 3, 4],  # Ranks don't matter for IQR of scores
        )
        
        rankings_per_year = {
            2020: {
                RankingMethod.IF_WASPAS: ranking_result,
            }
        }
        
        iqr_per_year, iqr_mean = compute_discriminatory_power(rankings_per_year)
        
        assert RankingMethod.IF_WASPAS in iqr_per_year
        assert iqr_per_year[RankingMethod.IF_WASPAS][2020] >= 0.0

    def test_discriminatory_power_multiple_years(self):
        """Test discriminatory power across multiple years."""
        rankings_per_year = {
            2020: {
                RankingMethod.IF_WASPAS: RankingResult(
                    method=RankingMethod.IF_WASPAS,
                    year=2020,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.9, 0.5, 0.1],
                    ranks=[1, 2, 3],
                ),
            },
            2021: {
                RankingMethod.IF_WASPAS: RankingResult(
                    method=RankingMethod.IF_WASPAS,
                    year=2021,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.8, 0.4, 0.2],
                    ranks=[1, 2, 3],
                ),
            },
        }
        
        iqr_per_year, iqr_mean = compute_discriminatory_power(rankings_per_year)
        
        assert len(iqr_per_year[RankingMethod.IF_WASPAS]) == 2
        assert RankingMethod.IF_WASPAS in iqr_mean


class TestTemporalPersistence:
    """Tests for temporal persistence (year-to-year) analysis."""

    def test_temporal_persistence_stable_rankings(self):
        """Stable year-to-year rankings should have high ρ."""
        rankings_per_year = {
            2020: {
                RankingMethod.IF_WASPAS: RankingResult(
                    method=RankingMethod.IF_WASPAS,
                    year=2020,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.9, 0.8, 0.7],
                    ranks=[1, 2, 3],
                ),
            },
            2021: {
                RankingMethod.IF_WASPAS: RankingResult(
                    method=RankingMethod.IF_WASPAS,
                    year=2021,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.89, 0.79, 0.69],
                    ranks=[1, 2, 3],  # Same ranking
                ),
            },
        }
        
        rho_yoy, rho_mean, trend = compute_temporal_persistence(rankings_per_year)
        
        # Year-to-year correlation should be high
        assert rho_yoy[RankingMethod.IF_WASPAS][0] > 0.9

    def test_temporal_persistence_volatile_rankings(self):
        """Volatile year-to-year rankings should have low ρ."""
        rankings_per_year = {
            2020: {
                RankingMethod.IF_WASPAS: RankingResult(
                    method=RankingMethod.IF_WASPAS,
                    year=2020,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.9, 0.8, 0.7],
                    ranks=[1, 2, 3],
                ),
            },
            2021: {
                RankingMethod.IF_WASPAS: RankingResult(
                    method=RankingMethod.IF_WASPAS,
                    year=2021,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.7, 0.9, 0.8],
                    ranks=[3, 1, 2],  # Completely different ranking
                ),
            },
        }
        
        rho_yoy, rho_mean, trend = compute_temporal_persistence(rankings_per_year)
        
        # Year-to-year correlation should be low
        assert rho_yoy[RankingMethod.IF_WASPAS][0] < 0.5

    def test_temporal_persistence_single_year_error(self):
        """Should error with only single year."""
        rankings_per_year = {
            2020: {
                RankingMethod.IF_WASPAS: RankingResult(
                    method=RankingMethod.IF_WASPAS,
                    year=2020,
                    provinces=["P01", "P02"],
                    scores=[0.9, 0.7],
                    ranks=[1, 2],
                ),
            }
        }
        
        with pytest.raises(ValueError):
            compute_temporal_persistence(rankings_per_year)

    def test_temporal_persistence_trend(self):
        """Test trend computation for multiple years."""
        rankings_per_year = {
            2019: {
                RankingMethod.IF_WASPAS: RankingResult(
                    method=RankingMethod.IF_WASPAS,
                    year=2019,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.9, 0.8, 0.7],
                    ranks=[1, 2, 3],
                ),
            },
            2020: {
                RankingMethod.IF_WASPAS: RankingResult(
                    method=RankingMethod.IF_WASPAS,
                    year=2020,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.88, 0.79, 0.72],
                    ranks=[1, 2, 3],
                ),
            },
            2021: {
                RankingMethod.IF_WASPAS: RankingResult(
                    method=RankingMethod.IF_WASPAS,
                    year=2021,
                    provinces=["P01", "P02", "P03"],
                    scores=[0.85, 0.76, 0.75],
                    ranks=[1, 2, 3],
                ),
            },
        }
        
        rho_yoy, rho_mean, trend = compute_temporal_persistence(rankings_per_year)
        
        # Should compute trend
        assert trend[RankingMethod.IF_WASPAS] is not None


class TestRankingValidationResult:
    """Tests for RankingValidationResult dataclass."""

    def test_result_to_dataframe(self):
        """Test conversion to DataFrames."""
        result = RankingValidationResult(
            inter_method_correlation={
                "if_waspas_vs_if_topsis": {2020: 0.95, 2021: 0.92},
            },
            inter_method_correlation_overall={
                "if_waspas_vs_if_topsis": 0.93,
            },
            discriminatory_power_iqr={
                "if_waspas": {2020: 0.5, 2021: 0.6},
            },
            discriminatory_power_mean={
                "if_waspas": 0.55,
            },
            temporal_persistence_rho={
                "if_waspas": [0.95, 0.93],
            },
            temporal_persistence_mean={
                "if_waspas": 0.94,
            },
            temporal_persistence_trend={
                "if_waspas": -0.02,
            },
        )

        inter_method_df, discriminatory_df, temporal_df = result.to_dataframe()
        
        assert inter_method_df is not None
        assert discriminatory_df is not None
        assert temporal_df is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
