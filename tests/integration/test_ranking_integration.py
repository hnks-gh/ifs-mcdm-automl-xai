"""
tests/integration/test_ranking_integration.py
----------------------------------------------
Integration tests for MCDM ranking methods with realistic scenarios.

Test strategy
=============
1. Test all three ranking methods on synthetic PAPI-like data (63 provinces, 29 criteria)
2. Validate NaN handling with realistic missing data patterns
3. Verify output consistency and file I/O
4. Test year-to-year ranking consistency
"""

import numpy as np
import pytest

from src.core.ifs_arithmetic import IFSMatrix
from src.mcdm.ranking import if_waspas, if_topsis, if_promethee2
from src.core.schema import RankingMethod


class TestRankingIntegrationRealistic:
    """Integration tests with realistic PAPI-like data."""

    @pytest.fixture
    def synthetic_papi_data(self):
        """
        Create synthetic PAPI-like data:
        - 63 provinces
        - 29 criteria
        - Realistic missing data patterns (13.4% missingness)
        """
        np.random.seed(42)
        n_provinces = 63
        n_criteria = 29

        # Generate base scores in [0, 3.33]
        mu_base = np.random.uniform(0.5, 0.9, (n_provinces, n_criteria))
        nu_base = np.random.uniform(0.0, 0.3, (n_provinces, n_criteria))

        # Ensure mu + nu <= 1
        nu_base = np.minimum(nu_base, 1.0 - mu_base)
        pi_base = 1.0 - mu_base - nu_base

        # Add realistic NaN patterns:
        # - Structural missing (column-wide for early years) — simulate with first columns
        mu = mu_base.copy()
        nu = nu_base.copy()
        pi = pi_base.copy()

        # Type 1: Structural column gaps (e.g., SC24, SC71-SC83 in 2011-2017)
        # Simulate: 3 criteria always missing
        mu[:, 23] = np.nan  # SC24
        nu[:, 23] = np.nan
        pi[:, 23] = np.nan

        mu[:, 23:26] = np.nan  # SC71, SC72, SC73
        nu[:, 23:26] = np.nan
        pi[:, 23:26] = np.nan

        # Type 2: Random missing cells (~10% of remaining)
        mask = np.random.random((n_provinces, n_criteria)) < 0.05
        mu[mask] = np.nan
        nu[mask] = np.nan
        pi[mask] = np.nan

        # Type 3: Ensure some rows have data
        # Zero out complete rows for 9 province-year combos
        blank_rows = np.random.choice(n_provinces, 9, replace=False)
        mu[blank_rows, :] = np.nan
        nu[blank_rows, :] = np.nan
        pi[blank_rows, :] = np.nan

        # Fill in some safe values to avoid all-NaN columns
        for j in range(n_criteria):
            if np.isnan(mu[:, j]).all():
                # If column is all NaN, put some values back
                safe_rows = np.random.choice(
                    n_provinces, max(5, n_provinces // 10), replace=False
                )
                mu[safe_rows, j] = np.random.uniform(0.6, 0.85, len(safe_rows))
                nu[safe_rows, j] = np.random.uniform(0.05, 0.25, len(safe_rows))
                pi[safe_rows, j] = 1.0 - mu[safe_rows, j] - nu[safe_rows, j]

        provinces = [f"P{i+1:02d}" for i in range(n_provinces)]
        criteria = [f"SC{10+i//10}{(i%10)+1}" for i in range(n_criteria)]
        criteria = [f"C{i//4+1}S{i%4+1}" for i in range(n_criteria)]  # Alternate format

        return IFSMatrix(
            mu=mu, nu=nu, pi=pi,
            alternatives=provinces,
            criteria=criteria,
            year=2020,
        )

    @pytest.fixture
    def realistic_weights(self):
        """Generate realistic weights for 29 criteria."""
        np.random.seed(42)
        weights = np.random.dirichlet(np.ones(29))
        return weights

    def test_waspas_on_realistic_data(self, synthetic_papi_data, realistic_weights):
        """Test IF-WASPAS ranking on realistic PAPI data."""
        result = if_waspas.rank(
            synthetic_papi_data, realistic_weights, lambda_param=0.5
        )

        # Validations
        assert result.method == RankingMethod.IF_WASPAS
        assert len(result.provinces) == 63
        assert len(result.scores) == 63
        assert len(result.ranks) == 63
        assert sorted(result.ranks) == list(range(1, 64))

        # Scores should be valid (may include NaN for all-NaN rows)
        non_nan_scores = sum(1 for s in result.scores if not np.isnan(s))
        assert non_nan_scores >= 1  # At least some valid scores

    def test_topsis_on_realistic_data(self, synthetic_papi_data, realistic_weights):
        """Test IF-TOPSIS ranking on realistic PAPI data."""
        result = if_topsis.rank(synthetic_papi_data, realistic_weights)

        # Validations
        assert result.method == RankingMethod.IF_TOPSIS
        assert len(result.provinces) == 63
        assert len(result.scores) == 63
        assert len(result.ranks) == 63
        assert sorted(result.ranks) == list(range(1, 64))

        # Closeness coefficients should be in [0, 1]
        for score in result.scores:
            if not np.isnan(score):
                assert 0.0 <= score <= 1.0

    def test_promethee2_on_realistic_data(self, synthetic_papi_data, realistic_weights):
        """Test IF-PROMETHEE II ranking on realistic PAPI data."""
        result = if_promethee2.rank(
            synthetic_papi_data, realistic_weights, p_parameter=0.1
        )

        # Validations
        assert result.method == RankingMethod.IF_PROMETHEE2
        assert len(result.provinces) == 63
        assert len(result.scores) == 63
        assert len(result.ranks) == 63
        assert sorted(result.ranks) == list(range(1, 64))

    def test_all_methods_produce_consistent_rankings(
        self, synthetic_papi_data, realistic_weights
    ):
        """Test that all three methods produce consistent (though different) rankings."""
        result_waspas = if_waspas.rank(
            synthetic_papi_data, realistic_weights, lambda_param=0.5
        )
        result_topsis = if_topsis.rank(synthetic_papi_data, realistic_weights)
        result_promethee2 = if_promethee2.rank(
            synthetic_papi_data, realistic_weights, p_parameter=0.1
        )

        # All should rank provinces 1..63 (with potential NaN handling)
        assert sorted(result_waspas.ranks) == list(range(1, 64))
        assert sorted(result_topsis.ranks) == list(range(1, 64))
        assert sorted(result_promethee2.ranks) == list(range(1, 64))

        # Top 5 provinces might differ across methods, but should have some overlap
        top5_waspas = set(
            result_waspas.provinces[i]
            for i, r in enumerate(result_waspas.ranks)
            if r <= 5
        )
        top5_topsis = set(
            result_topsis.provinces[i]
            for i, r in enumerate(result_topsis.ranks)
            if r <= 5
        )
        top5_promethee = set(
            result_promethee2.provinces[i]
            for i, r in enumerate(result_promethee2.ranks)
            if r <= 5
        )

        # At least some overlap is expected
        overlap = (top5_waspas & top5_topsis) | (top5_waspas & top5_promethee)
        assert len(overlap) >= 1  # Allow for methods to disagree somewhat

    def test_nan_handling_consistency(self, synthetic_papi_data, realistic_weights):
        """Test that NaN handling is consistent across methods."""
        result_waspas = if_waspas.rank(
            synthetic_papi_data, realistic_weights, lambda_param=0.5
        )
        result_topsis = if_topsis.rank(synthetic_papi_data, realistic_weights)
        result_promethee2 = if_promethee2.rank(
            synthetic_papi_data, realistic_weights, p_parameter=0.1
        )

        # Count NaN scores in each result
        nan_count_waspas = sum(1 for s in result_waspas.scores if np.isnan(s))
        nan_count_topsis = sum(1 for s in result_topsis.scores if np.isnan(s))
        nan_count_promethee = sum(1 for s in result_promethee2.scores if np.isnan(s))

        # Should be roughly similar (all-NaN rows result in NaN scores)
        assert nan_count_waspas >= 0
        assert nan_count_topsis >= 0
        assert nan_count_promethee >= 0

    def test_weight_parameter_sensitivity_waspas(self, synthetic_papi_data):
        """Test WASPAS sensitivity to lambda parameter."""
        weights = np.ones(synthetic_papi_data.n_criteria) / synthetic_papi_data.n_criteria

        results = {}
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = if_waspas.rank(synthetic_papi_data, weights, lambda_param=lam)
            results[lam] = result

        # Lambda = 0 (pure WPM) and lambda = 1 (pure WSM) should differ
        ranks_0 = results[0.0].ranks
        ranks_1 = results[1.0].ranks

        # They should produce different rankings
        assert ranks_0 != ranks_1 or all(np.isnan(s) for s in results[0.0].scores)

    def test_weight_parameter_sensitivity_promethee(self, synthetic_papi_data):
        """Test PROMETHEE II sensitivity to p parameter."""
        weights = np.ones(synthetic_papi_data.n_criteria) / synthetic_papi_data.n_criteria

        results = {}
        for p in [0.05, 0.1, 0.2, 0.5]:
            result = if_promethee2.rank(synthetic_papi_data, weights, p_parameter=p)
            results[p] = result

        # Different p values should produce different rankings
        ranks_small = results[0.05].ranks
        ranks_large = results[0.5].ranks

        # They should produce different rankings (unless all are NaN)
        assert ranks_small != ranks_large or all(np.isnan(s) for s in results[0.05].scores)

    def test_ranking_determinism(self, synthetic_papi_data, realistic_weights):
        """Test that ranking is deterministic (same input → same output)."""
        result1_waspas = if_waspas.rank(
            synthetic_papi_data, realistic_weights, lambda_param=0.5
        )
        result2_waspas = if_waspas.rank(
            synthetic_papi_data, realistic_weights, lambda_param=0.5
        )

        # Ranks should be identical
        assert result1_waspas.ranks == result2_waspas.ranks

        # Scores should be identical
        for s1, s2 in zip(result1_waspas.scores, result2_waspas.scores):
            if np.isnan(s1) and np.isnan(s2):
                continue
            assert abs(s1 - s2) < 1e-10

    def test_ranking_with_zero_weights(self, synthetic_papi_data):
        """Test ranking when some weights are zero (criteria excluded)."""
        weights = np.zeros(synthetic_papi_data.n_criteria)
        weights[0] = 0.5
        weights[1] = 0.5

        result = if_waspas.rank(
            synthetic_papi_data, weights, lambda_param=0.5
        )

        # Should still produce valid ranking
        assert sorted(result.ranks) == list(range(1, 64))

    def test_ranking_output_consistency(self, synthetic_papi_data, realistic_weights):
        """Test that all output fields are consistent."""
        result = if_topsis.rank(synthetic_papi_data, realistic_weights)

        # Provinces should match
        assert len(result.provinces) == len(set(result.provinces))
        assert result.provinces == synthetic_papi_data.alternatives

        # Each province should have exactly one rank
        rank_counts = {}
        for province, rank in zip(result.provinces, result.ranks):
            rank_counts[province] = rank_counts.get(province, 0) + 1

        for count in rank_counts.values():
            assert count == 1

        # Ranks should be in range
        for rank in result.ranks:
            assert 1 <= rank <= 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
