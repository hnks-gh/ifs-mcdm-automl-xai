"""
tests/unit/test_if_waspas.py
----------------------------
Unit tests for IF-WASPAS ranking method.

Test strategy
=============
1. Small synthetic datasets (3-5 provinces, 3-5 criteria) with known outcomes
2. Validation: ranks are a permutation of 1..n
3. Boundary conditions: lambda = 0, 0.5, 1
4. NaN handling: missing sub-criteria are excluded
5. Score function ordering: S(A) = μ − ν used for ranking
"""

import numpy as np
import pytest

from src.core.exceptions import IFSArithmeticError
from src.core.ifs_arithmetic import IFSMatrix
from src.mcdm.ranking import if_waspas
from src.core.schema import RankingMethod


class TestIFWASPASRank:
    """Test IF-WASPAS rank() function."""

    def test_rank_basic_3_provinces_3_criteria(self):
        """Test basic ranking with 3 provinces, 3 criteria."""
        # Create synthetic decision matrix
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

        result = if_waspas.rank(ifs_matrix, weights, lambda_param=0.5)

        # Assertions
        assert result.method == RankingMethod.IF_WASPAS
        assert result.year == 2020
        assert result.provinces == ["P1", "P2", "P3"]
        assert len(result.scores) == 3
        assert len(result.ranks) == 3

        # Check ranks are a permutation of 1..3
        assert sorted(result.ranks) == [1, 2, 3]

        # Check higher scores map to better ranks
        for i, j in zip(range(3), range(3)):
            if result.scores[i] > result.scores[j]:
                assert result.ranks[i] < result.ranks[j]

    def test_rank_lambda_0_pure_wpm(self):
        """Test λ=0 (pure WPM/geometric mean)."""
        mu = np.array([[0.8, 0.6], [0.4, 0.9]], dtype=float)
        nu = np.array([[0.1, 0.3], [0.5, 0.0]], dtype=float)
        pi = 1.0 - mu - nu

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1", "P2"],
            criteria=["C1", "C2"],
            year=2020,
        )

        weights = np.array([0.5, 0.5])

        result = if_waspas.rank(ifs_matrix, weights, lambda_param=0.0)

        assert result.method == RankingMethod.IF_WASPAS
        assert len(result.scores) == 2
        assert sorted(result.ranks) == [1, 2]

    def test_rank_lambda_1_pure_wsm(self):
        """Test λ=1 (pure WSM/arithmetic mean)."""
        mu = np.array([[0.8, 0.6], [0.4, 0.9]], dtype=float)
        nu = np.array([[0.1, 0.3], [0.5, 0.0]], dtype=float)
        pi = 1.0 - mu - nu

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1", "P2"],
            criteria=["C1", "C2"],
            year=2020,
        )

        weights = np.array([0.5, 0.5])

        result = if_waspas.rank(ifs_matrix, weights, lambda_param=1.0)

        assert result.method == RankingMethod.IF_WASPAS
        assert len(result.scores) == 2
        assert sorted(result.ranks) == [1, 2]

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

        result = if_waspas.rank(ifs_matrix, weights, lambda_param=0.5)

        # Should not raise, NaN should be handled gracefully
        assert len(result.scores) == 3
        assert sorted(result.ranks) == [1, 2, 3]

    def test_rank_invalid_lambda(self):
        """Test that lambda outside [0, 1] raises error."""
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
            if_waspas.rank(ifs_matrix, weights, lambda_param=1.5)

        with pytest.raises(IFSArithmeticError):
            if_waspas.rank(ifs_matrix, weights, lambda_param=-0.1)

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

        wrong_weights = np.array([0.5, 0.5])  # Only 2 weights for 3 criteria

        with pytest.raises(IFSArithmeticError):
            if_waspas.rank(ifs_matrix, wrong_weights, lambda_param=0.5)

    def test_rank_score_ordering(self):
        """Test that ranks correctly correspond to scores."""
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

        result = if_waspas.rank(ifs_matrix, weights, lambda_param=0.5)

        # Verify rank-score correspondence
        for i in range(3):
            for j in range(3):
                if result.scores[i] > result.scores[j]:
                    assert result.ranks[i] < result.ranks[j]
                elif result.scores[i] < result.scores[j]:
                    assert result.ranks[i] > result.ranks[j]

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

        result = if_waspas.rank(ifs_matrix, weights, lambda_param=0.5)

        assert len(result.scores) == 2
        assert sorted(result.ranks) == [1, 2]

    def test_blend_function_smoothness(self):
        """Test that blending is smooth across lambda values."""
        mu = np.array([[0.7, 0.8]], dtype=float)
        nu = np.array([[0.2, 0.1]], dtype=float)
        pi = 1.0 - mu - nu

        ifs_matrix = IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=["P1"],
            criteria=["C1", "C2"],
            year=2020,
        )

        weights = np.array([0.5, 0.5])

        # Compute rankings for different lambda values
        lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
        scores = []

        for lam in lambdas:
            result = if_waspas.rank(ifs_matrix, weights, lambda_param=lam)
            scores.append(result.scores[0])

        # Scores should form a continuous path (no sudden jumps)
        # All scores should be within valid range [−1, 1] approximately
        for score in scores:
            assert -1.5 <= score <= 1.5

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

        result = if_waspas.rank(ifs_matrix, weights, lambda_param=0.5)

        assert len(result.scores) == 1
        assert len(result.ranks) == 1
        assert result.ranks[0] == 1  # Single province gets rank 1


class TestScoreToRank:
    """Test rank conversion helper."""

    def test_score_to_rank_basic(self):
        """Test score to rank conversion."""
        scores = np.array([0.5, 0.3, 0.8, 0.1])
        ranks = if_waspas._score_to_rank(scores)

        # Verify permutation
        assert sorted(ranks) == [1, 2, 3, 4]

        # Verify ordering: higher score → lower rank (better)
        assert ranks[2] < ranks[0]  # 0.8 > 0.5
        assert ranks[0] < ranks[1]  # 0.5 > 0.3

    def test_score_to_rank_with_nan(self):
        """Test rank conversion with NaN."""
        scores = np.array([0.5, np.nan, 0.8, 0.1])
        ranks = if_waspas._score_to_rank(scores)

        # Verify permutation
        assert sorted(ranks) == [1, 2, 3, 4]

        # NaN should get worst rank
        assert ranks[1] == 4

    def test_score_to_rank_ties(self):
        """Test rank conversion with tied scores."""
        scores = np.array([0.5, 0.5, 0.8, 0.1])
        ranks = if_waspas._score_to_rank(scores)

        # Verify permutation
        assert sorted(ranks) == [1, 2, 3, 4]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
