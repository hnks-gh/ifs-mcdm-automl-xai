"""
tests/unit/test_if_topsis.py
----------------------------
Unit tests for IF-TOPSIS ranking method.

Test strategy
=============
1. Synthetic datasets (3-5 provinces, 3-5 criteria) with known ideal solutions
2. Validation: closeness coefficients in [0, 1]
3. Validation: ranks are a permutation of 1..n
4. Ideal point computation: PIS = max(μ), min(ν); NIS = min(μ), max(ν)
5. Distance calculations: normalized Euclidean distance
6. NaN handling: missing sub-criteria excluded via weight renormalisation
7. Weight application: scalar multiplication
"""

import numpy as np
import pytest

from src.core.exceptions import IFSArithmeticError
from src.core.ifs_arithmetic import IFSMatrix
from src.mcdm.ranking import if_topsis
from src.core.schema import RankingMethod


class TestIFTOPSISRank:
    """Test IF-TOPSIS rank() function."""

    def test_rank_basic_3_provinces_3_criteria(self):
        """Test basic ranking with 3 provinces, 3 criteria."""
        mu = np.array([
            [0.8, 0.7, 0.6],
            [0.5, 0.8, 0.7],
            [0.6, 0.5, 0.8],
        ], dtype=float)
        nu = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.1, 0.2],
            [0.3, 0.4, 0.1],
        ], dtype=float)
        pi = 1.0 - mu - nu

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1", "P2", "P3"],
            criteria=["SC1", "SC2", "SC3"],
            year=2020,
        )

        weights = np.array([0.4, 0.3, 0.3])

        result = if_topsis.rank(ifs_matrix, weights)

        # Assertions
        assert result.method == RankingMethod.IF_TOPSIS
        assert result.year == 2020
        assert result.provinces == ["P1", "P2", "P3"]
        assert len(result.scores) == 3
        assert len(result.ranks) == 3

        # Check ranks are a permutation of 1..3
        assert sorted(result.ranks) == [1, 2, 3]

        # Check closeness coefficients in [0, 1]
        for score in result.scores:
            assert 0.0 <= score <= 1.0 or np.isnan(score)

    def test_rank_closeness_coefficient_bounds(self):
        """Test that closeness coefficients are bounded in [0, 1]."""
        mu = np.array([
            [0.9, 0.8],
            [0.1, 0.2],
            [0.5, 0.5],
        ], dtype=float)
        nu = np.array([
            [0.0, 0.1],
            [0.8, 0.7],
            [0.4, 0.4],
        ], dtype=float)
        pi = 1.0 - mu - nu

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1", "P2", "P3"],
            criteria=["C1", "C2"],
            year=2020,
        )

        weights = np.array([0.5, 0.5])

        result = if_topsis.rank(ifs_matrix, weights)

        for score in result.scores:
            if not np.isnan(score):
                assert 0.0 <= score <= 1.0

    def test_rank_ideal_province_best(self):
        """Test that ideally best province ranks best."""
        # P1 is best: max μ and min ν
        mu = np.array([
            [1.0, 1.0],   # P1: ideal
            [0.0, 0.0],   # P2: worst
            [0.5, 0.5],   # P3: middle
        ], dtype=float)
        nu = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ], dtype=float)
        pi = 1.0 - mu - nu

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1", "P2", "P3"],
            criteria=["C1", "C2"],
            year=2020,
        )

        weights = np.array([0.5, 0.5])

        result = if_topsis.rank(ifs_matrix, weights)

        # P1 should rank 1 (best)
        p1_idx = result.provinces.index("P1")
        assert result.ranks[p1_idx] == 1

        # P2 should rank 3 (worst)
        p2_idx = result.provinces.index("P2")
        assert result.ranks[p2_idx] == 3

    def test_rank_with_nan_values(self):
        """Test ranking with NaN (missing) sub-criteria."""
        mu = np.array([
            [0.8, np.nan, 0.6],
            [0.5, 0.8, 0.7],
            [0.6, 0.5, np.nan],
        ], dtype=float)
        nu = np.array([
            [0.1, np.nan, 0.3],
            [0.4, 0.1, 0.2],
            [0.3, 0.4, np.nan],
        ], dtype=float)
        pi = np.array([
            [0.1, np.nan, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, np.nan],
        ], dtype=float)

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1", "P2", "P3"],
            criteria=["SC1", "SC2", "SC3"],
            year=2020,
        )

        weights = np.array([0.3, 0.3, 0.4])

        result = if_topsis.rank(ifs_matrix, weights)

        # Should not raise, NaN should be handled gracefully
        assert len(result.scores) == 3
        assert sorted(result.ranks) == [1, 2, 3]

    def test_rank_weight_length_mismatch(self):
        """Test that mismatched weight length raises error."""
        mu = np.array([[0.8, 0.6, 0.5]], dtype=float)
        nu = np.array([[0.1, 0.3, 0.4]], dtype=float)
        pi = 1.0 - mu - nu

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1"],
            criteria=["C1", "C2", "C3"],
            year=2020,
        )

        wrong_weights = np.array([0.5, 0.5])

        with pytest.raises(IFSArithmeticError):
            if_topsis.rank(ifs_matrix, wrong_weights)

    def test_rank_score_ordering(self):
        """Test that ranks correctly correspond to closeness."""
        mu = np.array([
            [0.9, 0.8],
            [0.5, 0.5],
            [0.7, 0.6],
        ], dtype=float)
        nu = np.array([
            [0.0, 0.1],
            [0.4, 0.4],
            [0.2, 0.3],
        ], dtype=float)
        pi = 1.0 - mu - nu

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1", "P2", "P3"],
            criteria=["C1", "C2"],
            year=2020,
        )

        weights = np.array([0.5, 0.5])

        result = if_topsis.rank(ifs_matrix, weights)

        # Verify rank-score correspondence
        for i in range(3):
            for j in range(3):
                if result.scores[i] > result.scores[j]:
                    assert result.ranks[i] < result.ranks[j]
                elif result.scores[i] < result.scores[j]:
                    assert result.ranks[i] > result.ranks[j]

    def test_rank_cost_criteria_parameter(self):
        """Test that cost_criteria parameter is accepted (not used)."""
        mu = np.array([[0.8, 0.6], [0.5, 0.8]], dtype=float)
        nu = np.array([[0.1, 0.3], [0.4, 0.1]], dtype=float)
        pi = 1.0 - mu - nu

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1", "P2"],
            criteria=["C1", "C2"],
            year=2020,
        )

        weights = np.array([0.5, 0.5])

        # Should accept cost_criteria without error (even if not used)
        result = if_topsis.rank(ifs_matrix, weights, cost_criteria=[])
        assert len(result.scores) == 2

    def test_rank_uniform_weights(self):
        """Test with uniform weights."""
        mu = np.array([
            [0.8, 0.6, 0.7],
            [0.5, 0.8, 0.6],
        ], dtype=float)
        nu = np.array([
            [0.1, 0.3, 0.2],
            [0.4, 0.1, 0.3],
        ], dtype=float)
        pi = 1.0 - mu - nu

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1", "P2"],
            criteria=["C1", "C2", "C3"],
            year=2020,
        )

        weights = np.array([1.0/3, 1.0/3, 1.0/3])

        result = if_topsis.rank(ifs_matrix, weights)

        assert len(result.scores) == 2
        assert sorted(result.ranks) == [1, 2]

    def test_rank_single_province(self):
        """Test ranking with a single province."""
        mu = np.array([[0.8, 0.6]], dtype=float)
        nu = np.array([[0.1, 0.3]], dtype=float)
        pi = 1.0 - mu - nu

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1"],
            criteria=["C1", "C2"],
            year=2020,
        )

        weights = np.array([0.5, 0.5])

        result = if_topsis.rank(ifs_matrix, weights)

        # Single province: closest to both PIS and NIS, CC should be ~0.5
        assert len(result.scores) == 1
        assert len(result.ranks) == 1
        assert result.ranks[0] == 1


class TestComputePISNIS:
    """Test PIS/NIS computation helpers."""

    def test_compute_pis_basic(self):
        """Test PIS computation."""
        mu = np.array([[0.9, 0.7], [0.5, 0.8]])
        nu = np.array([[0.0, 0.2], [0.4, 0.1]])
        pi = 1.0 - mu - nu

        pis_mu, pis_nu, pis_pi = if_topsis._compute_pis(mu, nu, pi)

        # PIS should have max μ and min ν per criterion
        np.testing.assert_array_almost_equal(pis_mu, [0.9, 0.8])
        np.testing.assert_array_almost_equal(pis_nu, [0.0, 0.1])

    def test_compute_nis_basic(self):
        """Test NIS computation."""
        mu = np.array([[0.9, 0.7], [0.5, 0.8]])
        nu = np.array([[0.0, 0.2], [0.4, 0.1]])
        pi = 1.0 - mu - nu

        nis_mu, nis_nu, nis_pi = if_topsis._compute_nis(mu, nu, pi)

        # NIS should have min μ and max ν per criterion
        np.testing.assert_array_almost_equal(nis_mu, [0.5, 0.7])
        np.testing.assert_array_almost_equal(nis_nu, [0.4, 0.2])


class TestApplyWeights:
    """Test weight application helper."""

    def test_apply_weights_basic(self):
        """Test IFS scalar multiplication for weights."""
        mu = np.array([[0.8, 0.6]])
        nu = np.array([[0.1, 0.3]])
        pi = 1.0 - mu - nu

        weights = np.array([0.5, 0.5])

        mu_w, nu_w, pi_w = if_topsis._apply_weights(mu, nu, pi, weights)

        # Result should have valid IFS values
        assert np.all(mu_w >= 0)
        assert np.all(nu_w >= 0)
        assert np.all(mu_w + nu_w <= 1.0 + 1e-10)

    def test_apply_weights_with_nan(self):
        """Test weight application with NaN."""
        mu = np.array([[0.8, np.nan]])
        nu = np.array([[0.1, np.nan]])
        pi = np.array([[0.1, np.nan]])

        weights = np.array([0.5, 0.5])

        mu_w, nu_w, pi_w = if_topsis._apply_weights(mu, nu, pi, weights)

        # NaN should be preserved
        assert np.isnan(mu_w[0, 1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
