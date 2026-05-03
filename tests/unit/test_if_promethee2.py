"""
tests/unit/test_if_promethee2.py
--------------------------------
Unit tests for IF-PROMETHEE II ranking method.

Test strategy
=============
1. Synthetic datasets (3-5 provinces, 3-5 criteria)
2. Validation: ranks are a permutation of 1..n
3. Preference function: Gaussian (0 for d≤0, 1-exp(-d²/(2p²)) for d>0)
4. Flow computation: positive, negative, and net flows
5. NaN handling: missing sub-criteria excluded via weight renormalisation
6. Preference is asymmetric: π(i,k) may differ from π(k,i)
"""

import numpy as np
import pytest

from src.core.exceptions import IFSArithmeticError
from src.core.ifs_arithmetic import IFSMatrix
from src.mcdm.ranking import if_promethee2
from src.core.schema import RankingMethod


class TestIFPROMETHEE2Rank:
    """Test IF-PROMETHEE II rank() function."""

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

        result = if_promethee2.rank(ifs_matrix, weights, p_parameter=0.1)

        # Assertions
        assert result.method == RankingMethod.IF_PROMETHEE2
        assert result.year == 2020
        assert result.provinces == ["P1", "P2", "P3"]
        assert len(result.scores) == 3
        assert len(result.ranks) == 3

        # Check ranks are a permutation of 1..3
        assert sorted(result.ranks) == [1, 2, 3]

    def test_rank_score_ordering(self):
        """Test that ranks correctly correspond to net flow."""
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

        result = if_promethee2.rank(ifs_matrix, weights, p_parameter=0.1)

        # Verify rank-score correspondence
        for i in range(3):
            for j in range(3):
                if result.scores[i] > result.scores[j]:
                    assert result.ranks[i] < result.ranks[j]
                elif result.scores[i] < result.scores[j]:
                    assert result.ranks[i] > result.ranks[j]

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

        result = if_promethee2.rank(ifs_matrix, weights, p_parameter=0.1)

        # Should not raise, NaN should be handled gracefully
        assert len(result.scores) == 3
        assert sorted(result.ranks) == [1, 2, 3]

    def test_rank_invalid_p_parameter(self):
        """Test that p_parameter ≤ 0 raises error."""
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

        with pytest.raises(IFSArithmeticError):
            if_promethee2.rank(ifs_matrix, weights, p_parameter=0.0)

        with pytest.raises(IFSArithmeticError):
            if_promethee2.rank(ifs_matrix, weights, p_parameter=-0.1)

    def test_rank_invalid_preference_function(self):
        """Test that invalid preference function raises error."""
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

        with pytest.raises(IFSArithmeticError):
            if_promethee2.rank(ifs_matrix, weights, preference_function="linear")

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
            if_promethee2.rank(ifs_matrix, wrong_weights)

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

        result = if_promethee2.rank(ifs_matrix, weights, p_parameter=0.1)

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

        result = if_promethee2.rank(ifs_matrix, weights, p_parameter=0.1)

        # Single province: φ⁺ = φ⁻ = 0, φ = 0
        assert len(result.scores) == 1
        assert len(result.ranks) == 1
        assert result.ranks[0] == 1
        assert result.scores[0] == 0.0  # Net flow is 0 for single province

    def test_rank_perfect_vs_worst(self):
        """Test that perfect province ranks better than worst province."""
        # P1 has high scores, P2 has low scores
        mu = np.array([
            [0.9, 0.9],
            [0.1, 0.1],
        ], dtype=float)
        nu = np.array([
            [0.0, 0.0],
            [0.8, 0.8],
        ], dtype=float)
        pi = 1.0 - mu - nu

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1", "P2"],
            criteria=["C1", "C2"],
            year=2020,
        )

        weights = np.array([0.5, 0.5])

        result = if_promethee2.rank(ifs_matrix, weights, p_parameter=0.1)

        # P1 should rank 1
        p1_idx = result.provinces.index("P1")
        assert result.ranks[p1_idx] == 1

        # P2 should rank 2
        p2_idx = result.provinces.index("P2")
        assert result.ranks[p2_idx] == 2


class TestPreferenceGaussian:
    """Test Gaussian preference function."""

    def test_gaussian_zero_or_negative_difference(self):
        """Test that preference is 0 for d ≤ 0."""
        p = 0.1
        assert if_promethee2.preference_gaussian(0.0, p) == 0.0
        assert if_promethee2.preference_gaussian(-0.1, p) == 0.0
        assert if_promethee2.preference_gaussian(-1.0, p) == 0.0

    def test_gaussian_positive_difference(self):
        """Test that preference increases with positive difference."""
        p = 0.1
        pref_1 = if_promethee2.preference_gaussian(0.01, p)
        pref_2 = if_promethee2.preference_gaussian(0.1, p)
        pref_3 = if_promethee2.preference_gaussian(0.5, p)

        # Preference increases with difference
        assert 0.0 < pref_1 < pref_2 < pref_3 < 1.0

    def test_gaussian_upper_bound(self):
        """Test that preference approaches 1 for large differences."""
        p = 0.1
        pref = if_promethee2.preference_gaussian(1.0, p)
        assert 0.99 <= pref <= 1.0

    def test_gaussian_symmetry_in_p(self):
        """Test Gaussian preference function symmetry properties."""
        p1 = 0.1
        p2 = 0.2
        d = 0.1

        pref_p1 = if_promethee2.preference_gaussian(d, p1)
        pref_p2 = if_promethee2.preference_gaussian(d, p2)

        # Larger p should give smaller preference for same d
        assert pref_p2 < pref_p1


class TestComputeFlows:
    """Test flow computation helpers."""

    def test_compute_flows_symmetric_case(self):
        """Test flow computation with symmetric preference matrix."""
        # Symmetric preference: all 0.5 off-diagonal
        pref_matrix = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ], dtype=float)

        phi_plus, phi_minus, phi_net = if_promethee2._compute_flows(pref_matrix)

        # All should have equal flows
        np.testing.assert_array_almost_equal(phi_plus, [0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(phi_minus, [0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(phi_net, [0.0, 0.0, 0.0])

    def test_compute_flows_asymmetric_case(self):
        """Test flow computation with asymmetric preference."""
        # P1 dominates: π(1,0) > π(0,1)
        pref_matrix = np.array([
            [0.0, 0.3, 0.2],
            [0.8, 0.0, 0.5],
            [0.7, 0.4, 0.0],
        ], dtype=float)

        phi_plus, phi_minus, phi_net = if_promethee2._compute_flows(pref_matrix)

        # P1 (row 1) has high outgoing preference → high φ⁺
        # P0 (row 0) has low outgoing preference → low φ⁺
        assert phi_plus[1] > phi_plus[0]

        # P0 has high incoming preference → high φ⁻
        # P1 has low incoming preference → low φ⁻
        assert phi_minus[0] > phi_minus[1]

    def test_compute_flows_single_province(self):
        """Test flow computation with single province."""
        pref_matrix = np.array([[0.0]], dtype=float)

        phi_plus, phi_minus, phi_net = if_promethee2._compute_flows(pref_matrix)

        assert phi_plus[0] == 0.0
        assert phi_minus[0] == 0.0
        assert phi_net[0] == 0.0


class TestComputePreferenceMatrix:
    """Test preference matrix computation."""

    def test_compute_preference_matrix_identical_scores(self):
        """Test preference when all alternatives have identical scores."""
        score_matrix = np.array([
            [0.5, 0.5],
            [0.5, 0.5],
        ], dtype=float)
        weights = np.array([0.5, 0.5])

        pref_matrix = if_promethee2._compute_preference_matrix(
            score_matrix, weights, p=0.1
        )

        # All preferences should be 0 (no differences)
        np.testing.assert_array_almost_equal(pref_matrix, np.zeros((2, 2)))

    def test_compute_preference_matrix_diagonal_zero(self):
        """Test that preference matrix has 0 diagonal (no self-preference)."""
        score_matrix = np.array([
            [0.9, 0.8],
            [0.5, 0.6],
            [0.7, 0.4],
        ], dtype=float)
        weights = np.array([0.5, 0.5])

        pref_matrix = if_promethee2._compute_preference_matrix(
            score_matrix, weights, p=0.1
        )

        # Diagonal should be all zeros
        np.testing.assert_array_almost_equal(np.diag(pref_matrix), [0.0, 0.0, 0.0])

    def test_compute_preference_matrix_with_nan(self):
        """Test preference matrix with NaN values."""
        score_matrix = np.array([
            [0.9, np.nan],
            [0.5, 0.6],
        ], dtype=float)
        weights = np.array([0.5, 0.5])

        pref_matrix = if_promethee2._compute_preference_matrix(
            score_matrix, weights, p=0.1
        )

        # Should handle NaN gracefully
        assert pref_matrix.shape == (2, 2)
        assert np.all(~np.isnan(pref_matrix))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
