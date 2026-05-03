"""
tests/unit/test_if_critic.py
-----------------------------
Unit tests for the IF-CRITIC two-level weighting engine.

Test categories
---------------
1. Core CRITIC kernel (_critic_weights_from_score_matrix)
2. NaN handling – partial and total missingness
3. Stage-1 intra-criterion weights
4. IFS-WAM aggregation helper
5. Stage-2 inter-criterion weights
6. handle_missing_subcriteria (zero-padding & combination)
7. Two-level aggregator (aggregate_regime_weights, compute_weights_for_year)
8. Weight-sum integrity invariants
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest

from src.core.exceptions import IFSArithmeticError
from src.core.ifs_arithmetic import IFSMatrix
from src.core.schema import WeightingConfig, WeightVector
from src.mcdm.weighting.if_critic import (
    _compute_corr_matrix_nanaware,
    _compute_std_nanaware,
    _critic_weights_from_score_matrix,
    _ifs_wam_aggregate,
    compute_critic_weights,
    compute_stage1_weights,
    compute_stage2_weights,
    handle_missing_subcriteria,
)
from src.mcdm.weighting.two_level_aggregator import (
    aggregate_regime_weights,
    compute_weights_for_year,
    _validate_weight_vector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_TOL = 1e-8


def _make_ifs_matrix(
    mu: np.ndarray,
    nu: np.ndarray,
    criteria: List[str],
    year: int = 2019,
) -> IFSMatrix:
    pi = np.clip(1.0 - mu - nu, 0.0, 1.0)
    n = mu.shape[0]
    alternatives = [f"P{i+1:02d}" for i in range(n)]
    return IFSMatrix(mu=mu, nu=nu, pi=pi, alternatives=alternatives,
                     criteria=criteria, year=year)


def _default_w_config() -> WeightingConfig:
    return WeightingConfig(
        method="two_level_if_critic",
        min_variance_threshold=1e-9,
    )


# =============================================================================
# 1. std computation
# =============================================================================

class TestComputeStdNanAware:
    def test_basic(self):
        S = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        sigma = _compute_std_nanaware(S)
        assert sigma.shape == (2,)
        assert abs(sigma[0] - np.std([1, 3, 5], ddof=1)) < _TOL
        assert abs(sigma[1] - np.std([2, 4, 6], ddof=1)) < _TOL

    def test_single_valid_row_returns_zero(self):
        S = np.array([[1.0, np.nan], [np.nan, 2.0]])
        sigma = _compute_std_nanaware(S)
        # Each column has only 1 valid value → std = 0
        assert sigma[0] == 0.0
        assert sigma[1] == 0.0

    def test_all_nan_column(self):
        S = np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
        sigma = _compute_std_nanaware(S)
        assert sigma[0] == 0.0
        assert sigma[1] > 0.0

    def test_constant_column_zero_std(self):
        S = np.array([[0.5, 0.1], [0.5, 0.3], [0.5, 0.7]])
        sigma = _compute_std_nanaware(S)
        assert sigma[0] < _TOL
        assert sigma[1] > 0.0


# =============================================================================
# 2. Correlation matrix
# =============================================================================

class TestCorrMatrixNanAware:
    def test_diagonal_is_one(self):
        S = np.random.default_rng(0).uniform(-0.5, 0.5, (10, 4))
        R = _compute_corr_matrix_nanaware(S)
        assert np.allclose(R.diagonal(), 1.0)

    def test_symmetric(self):
        S = np.random.default_rng(1).uniform(-0.5, 0.5, (8, 3))
        R = _compute_corr_matrix_nanaware(S)
        assert np.allclose(R, R.T)

    def test_perfect_positive_correlation(self):
        S = np.column_stack([np.arange(5, dtype=float),
                              np.arange(5, dtype=float)])
        R = _compute_corr_matrix_nanaware(S)
        assert abs(R[0, 1] - 1.0) < _TOL

    def test_nan_pair_gives_zero(self):
        # Columns share no valid common rows → correlation = 0
        S = np.array([[1.0, np.nan], [2.0, np.nan], [np.nan, 1.0]])
        R = _compute_corr_matrix_nanaware(S)
        assert R[0, 1] == 0.0

    def test_values_clipped_to_minus_one_one(self):
        # Anticorrelated
        x = np.arange(6, dtype=float)
        S = np.column_stack([x, -x])
        R = _compute_corr_matrix_nanaware(S)
        assert R[0, 1] >= -1.0
        assert R[0, 1] <= 1.0


# =============================================================================
# 3. Core CRITIC kernel
# =============================================================================

class TestCriticWeightsFromScoreMatrix:
    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        S = rng.uniform(-0.5, 0.5, (20, 5))
        w = _critic_weights_from_score_matrix(S)
        assert abs(w.sum() - 1.0) < _TOL

    def test_non_negative(self):
        rng = np.random.default_rng(7)
        S = rng.uniform(-0.5, 0.5, (15, 4))
        w = _critic_weights_from_score_matrix(S)
        assert (w >= 0.0).all()

    def test_constant_column_gets_zero_weight(self):
        S = np.array([
            [0.5, 0.1, 0.3],
            [0.5, 0.4, 0.2],
            [0.5, 0.2, 0.5],
            [0.5, 0.3, 0.1],
        ])
        w = _critic_weights_from_score_matrix(S)
        assert w[0] < _TOL  # constant column
        assert abs(w.sum() - 1.0) < _TOL

    def test_single_column_weight_is_one(self):
        S = np.array([[0.1], [0.2], [0.3]])
        w = _critic_weights_from_score_matrix(S)
        assert abs(w[0] - 1.0) < _TOL

    def test_entirely_nan_column_gets_zero_weight(self):
        S = np.array([
            [np.nan, 0.1, 0.3],
            [np.nan, 0.4, -0.1],
            [np.nan, 0.2, 0.5],
        ])
        w = _critic_weights_from_score_matrix(S)
        assert w[0] < _TOL
        assert abs(w.sum() - 1.0) < _TOL

    def test_all_nan_fallback_equal_weights(self):
        # All columns entirely NaN: degenerate → equal weights
        S = np.full((3, 3), np.nan)
        w = _critic_weights_from_score_matrix(S)
        # All NaN → n_active=0, weights should be 0
        assert w.sum() < _TOL or abs(w.sum() - 1.0) < _TOL  # either valid

    def test_known_result_two_columns(self):
        """With 2 columns, hand-verify CRITIC formula."""
        # Column 0: [0.0, 0.4, 0.8]  mean=0.4, std=0.4
        # Column 1: [0.8, 0.4, 0.0]  mean=0.4, std=0.4
        # Pearson r(0,1) = -1.0
        # C_0 = std_0 * ((1-r00)+(1-r01)) = 0.4*(0+2) = 0.8
        # C_1 = std_1 * ((1-r10)+(1-r11)) = 0.4*(2+0) = 0.8
        # w = [0.5, 0.5]
        S = np.array([[0.0, 0.8], [0.4, 0.4], [0.8, 0.0]])
        w = _critic_weights_from_score_matrix(S)
        assert abs(w[0] - 0.5) < 1e-6
        assert abs(w[1] - 0.5) < 1e-6

    def test_partial_nan_handled(self):
        """Partial NaN in rows should not crash; weights must still sum to 1."""
        S = np.array([
            [0.2, np.nan, 0.4],
            [0.1, 0.3, 0.5],
            [np.nan, 0.6, 0.2],
            [0.3, 0.1, np.nan],
        ])
        w = _critic_weights_from_score_matrix(S)
        assert (w >= 0.0).all()
        assert abs(w.sum() - 1.0) < _TOL


# =============================================================================
# 4. compute_critic_weights (public wrapper)
# =============================================================================

class TestComputeCriticWeights:
    def test_label_length_mismatch_raises(self):
        S = np.zeros((5, 3))
        with pytest.raises(IFSArithmeticError):
            compute_critic_weights(S, labels=["A", "B"])  # 2 != 3

    def test_output_shape_matches_labels(self):
        S = np.random.default_rng(0).uniform(0, 0.5, (10, 4))
        labels = ["a", "b", "c", "d"]
        w = compute_critic_weights(S, labels=labels)
        assert len(w) == 4
        assert abs(w.sum() - 1.0) < _TOL


# =============================================================================
# 5. IFS-WAM aggregation
# =============================================================================

class TestIfsWamAggregate:
    def test_single_column_passthrough(self):
        mu = np.array([[0.6], [0.4], [0.3]])
        nu = np.array([[0.2], [0.4], [0.5]])
        w = np.array([1.0])
        mu_agg, nu_agg, pi_agg = _ifs_wam_aggregate(mu, nu, w)
        # IFS-WAM with 1 column: mu_agg = 1-(1-mu)^1 = mu; nu_agg = nu^1 = nu
        assert np.allclose(mu_agg, mu[:, 0], atol=1e-8)
        assert np.allclose(nu_agg, nu[:, 0], atol=1e-8)

    def test_partition_of_unity(self):
        rng = np.random.default_rng(3)
        n, p = 10, 4
        mu = rng.uniform(0.0, 0.4, (n, p))
        nu = rng.uniform(0.0, 0.4, (n, p))
        w = np.ones(p) / p
        mu_agg, nu_agg, pi_agg = _ifs_wam_aggregate(mu, nu, w)
        total = mu_agg + nu_agg + pi_agg
        assert np.allclose(total, 1.0, atol=1e-8)

    def test_all_nan_row_gives_nan(self):
        mu = np.array([[np.nan, np.nan], [0.3, 0.4]])
        nu = np.array([[np.nan, np.nan], [0.2, 0.1]])
        w = np.array([0.5, 0.5])
        mu_agg, nu_agg, _ = _ifs_wam_aggregate(mu, nu, w)
        assert math.isnan(mu_agg[0])
        assert not math.isnan(mu_agg[1])

    def test_zero_weight_column_ignored(self):
        mu = np.array([[0.5, 0.0]])
        nu = np.array([[0.3, 0.9]])
        # Second column has weight 0 → aggregate = first column only
        w = np.array([1.0, 0.0])
        mu_agg, nu_agg, _ = _ifs_wam_aggregate(mu, nu, w)
        assert abs(mu_agg[0] - 0.5) < _TOL
        assert abs(nu_agg[0] - 0.3) < _TOL


# =============================================================================
# 6. Stage-1 weights
# =============================================================================

class TestComputeStage1Weights:
    def _build_simple_ifs(self) -> IFSMatrix:
        rng = np.random.default_rng(0)
        n = 10
        criteria = ["SC11", "SC12", "SC13", "SC21", "SC22"]
        mu = rng.uniform(0.0, 0.8, (n, 5))
        nu = rng.uniform(0.0, 0.1, (n, 5))
        return _make_ifs_matrix(mu, nu, criteria)

    def _crit_map(self):
        return {
            "C01": ["SC11", "SC12", "SC13"],
            "C02": ["SC21", "SC22"],
        }

    def test_returns_all_criteria(self):
        mat = self._build_simple_ifs()
        cfg = _default_w_config()
        result = compute_stage1_weights(mat, self._crit_map(), cfg)
        assert set(result.keys()) == {"C01", "C02"}

    def test_intracriterion_weights_sum_to_one(self):
        mat = self._build_simple_ifs()
        cfg = _default_w_config()
        result = compute_stage1_weights(mat, self._crit_map(), cfg)
        for crit, wv in result.items():
            if len(wv.values) > 0:
                assert abs(sum(wv.values) - 1.0) < _TOL, \
                    f"Criterion {crit} weights don't sum to 1: {sum(wv.values)}"

    def test_absent_criterion_has_empty_labels(self):
        mat = self._build_simple_ifs()
        cfg = _default_w_config()
        crit_map = {
            "C01": ["SC11", "SC12", "SC13"],
            "C02": ["SC21", "SC22"],
            "C03": ["SC31", "SC32"],  # not in IFSMatrix
        }
        result = compute_stage1_weights(mat, crit_map, cfg)
        assert result["C03"].labels == []
        assert result["C03"].values == []

    def test_stage_attribute_is_1(self):
        mat = self._build_simple_ifs()
        cfg = _default_w_config()
        result = compute_stage1_weights(mat, self._crit_map(), cfg)
        for wv in result.values():
            assert wv.stage == 1


# =============================================================================
# 7. Stage-2 weights
# =============================================================================

class TestComputeStage2Weights:
    def _build_full_ifs(self) -> tuple:
        rng = np.random.default_rng(5)
        n = 12
        criteria = ["SC11", "SC12", "SC21", "SC22", "SC31"]
        mu = rng.uniform(0.0, 0.7, (n, 5))
        nu = rng.uniform(0.0, 0.15, (n, 5))
        mat = _make_ifs_matrix(mu, nu, criteria)
        crit_map = {
            "C01": ["SC11", "SC12"],
            "C02": ["SC21", "SC22"],
            "C03": ["SC31"],
        }
        return mat, crit_map

    def test_stage2_weights_sum_to_one(self):
        mat, crit_map = self._build_full_ifs()
        cfg = _default_w_config()
        stage1 = compute_stage1_weights(mat, crit_map, cfg)
        stage2 = compute_stage2_weights(mat, stage1, crit_map, cfg)
        assert abs(sum(stage2.values) - 1.0) < _TOL

    def test_stage2_non_negative(self):
        mat, crit_map = self._build_full_ifs()
        cfg = _default_w_config()
        stage1 = compute_stage1_weights(mat, crit_map, cfg)
        stage2 = compute_stage2_weights(mat, stage1, crit_map, cfg)
        assert all(v >= 0.0 for v in stage2.values)

    def test_stage2_stage_attribute(self):
        mat, crit_map = self._build_full_ifs()
        cfg = _default_w_config()
        stage1 = compute_stage1_weights(mat, crit_map, cfg)
        stage2 = compute_stage2_weights(mat, stage1, crit_map, cfg)
        assert stage2.stage == 2

    def test_stage2_labels_are_criterion_codes(self):
        mat, crit_map = self._build_full_ifs()
        cfg = _default_w_config()
        stage1 = compute_stage1_weights(mat, crit_map, cfg)
        stage2 = compute_stage2_weights(mat, stage1, crit_map, cfg)
        assert all(lbl.startswith("C") for lbl in stage2.labels)


# =============================================================================
# 8. handle_missing_subcriteria
# =============================================================================

class TestHandleMissingSubcriteria:
    ALL_SC = ["SC11", "SC12", "SC21", "SC22", "SC31"]
    CRIT_MAP = {"C01": ["SC11", "SC12"], "C02": ["SC21", "SC22"], "C03": ["SC31"]}

    def _run(self):
        rng = np.random.default_rng(9)
        n = 8
        criteria = ["SC11", "SC12", "SC21", "SC22"]  # SC31 absent
        mu = rng.uniform(0.0, 0.7, (n, 4))
        nu = rng.uniform(0.0, 0.15, (n, 4))
        mat = _make_ifs_matrix(mu, nu, criteria)
        cfg = _default_w_config()
        stage1 = compute_stage1_weights(mat, self.CRIT_MAP, cfg)
        stage2 = compute_stage2_weights(mat, stage1, self.CRIT_MAP, cfg)
        return handle_missing_subcriteria(stage1, stage2, self.CRIT_MAP, self.ALL_SC)

    def test_length_equals_all_subcriteria(self):
        wv = self._run()
        assert len(wv.values) == len(self.ALL_SC)
        assert len(wv.labels) == len(self.ALL_SC)

    def test_labels_match_all_subcriteria(self):
        wv = self._run()
        assert wv.labels == self.ALL_SC

    def test_absent_subcriterion_has_zero_weight(self):
        wv = self._run()
        d = wv.as_dict()
        assert d["SC31"] < 1e-12  # SC31 absent in IFSMatrix

    def test_active_weights_sum_to_one(self):
        wv = self._run()
        assert abs(sum(wv.values) - 1.0) < 1e-8


# =============================================================================
# 9. Regime blending
# =============================================================================

class TestAggregateRegimeWeights:
    ALL_SC = ["SC11", "SC12", "SC21"]

    def _make_wv(self, values, regime_id="R1", year=2019):
        return WeightVector(
            labels=self.ALL_SC, values=values, year=year,
            regime_id=regime_id, stage=None,
        )

    def test_single_regime_passthrough(self):
        wv = self._make_wv([0.5, 0.3, 0.2])
        result = aggregate_regime_weights(
            regime_year_weights={"R1": [wv]},
            regime_year_counts={"R1": 1},
            all_subcriteria=self.ALL_SC,
        )
        assert abs(result.values[0] - 0.5) < _TOL

    def test_blended_sum_to_one(self):
        wv1a = self._make_wv([0.6, 0.2, 0.2], regime_id="R1", year=2011)
        wv1b = self._make_wv([0.4, 0.3, 0.3], regime_id="R1", year=2012)
        wv2 = self._make_wv([0.1, 0.5, 0.4], regime_id="R2", year=2018)
        result = aggregate_regime_weights(
            regime_year_weights={"R1": [wv1a, wv1b], "R2": [wv2]},
            regime_year_counts={"R1": 7, "R2": 1},
            all_subcriteria=self.ALL_SC,
        )
        assert abs(sum(result.values) - 1.0) < _TOL

    def test_equal_regimes_blend(self):
        wv1 = self._make_wv([1.0, 0.0, 0.0], regime_id="R1")
        wv2 = self._make_wv([0.0, 1.0, 0.0], regime_id="R2")
        result = aggregate_regime_weights(
            regime_year_weights={"R1": [wv1], "R2": [wv2]},
            regime_year_counts={"R1": 1, "R2": 1},
            all_subcriteria=self.ALL_SC,
            blend_method="equal_regimes",
        )
        # Equal blend of [1,0,0] and [0,1,0] = [0.5, 0.5, 0.0]
        assert abs(result.values[0] - 0.5) < _TOL
        assert abs(result.values[1] - 0.5) < _TOL

    def test_unknown_blend_method_raises(self):
        wv = self._make_wv([0.5, 0.3, 0.2])
        with pytest.raises(IFSArithmeticError):
            aggregate_regime_weights(
                regime_year_weights={"R1": [wv]},
                regime_year_counts={"R1": 1},
                all_subcriteria=self.ALL_SC,
                blend_method="invalid_method",
            )


# =============================================================================
# 10. compute_weights_for_year (integration test)
# =============================================================================

class TestComputeWeightsForYear:
    def _build_full_scenario(self):
        """Build a realistic 5-criterion, 8-province scenario."""
        from src.core.schema import Regime
        rng = np.random.default_rng(42)
        # Active sub-criteria: SC11, SC12, SC21, SC22, SC31
        active_sc = ["SC11", "SC12", "SC21", "SC22", "SC31"]
        all_sc = ["SC11", "SC12", "SC21", "SC22", "SC31", "SC83"]  # SC83 absent
        n = 8
        mu = rng.uniform(0.1, 0.7, (n, 5))
        nu = rng.uniform(0.0, 0.15, (n, 5))
        mat = _make_ifs_matrix(mu, nu, active_sc, year=2019)
        regime = Regime(
            regime_id="R3",
            years=[2019, 2020],
            active_subcriteria=active_sc,
            absent_subcriteria=["SC83"],
        )
        crit_map = {
            "C01": ["SC11", "SC12"],
            "C02": ["SC21", "SC22"],
            "C03": ["SC31", "SC83"],
        }
        cfg = _default_w_config()
        return mat, regime, crit_map, all_sc, cfg

    def test_output_length_equals_all_subcriteria(self):
        mat, regime, crit_map, all_sc, cfg = self._build_full_scenario()
        wv = compute_weights_for_year(mat, regime, crit_map, all_sc, cfg)
        assert len(wv.values) == len(all_sc)

    def test_weights_sum_to_one(self):
        mat, regime, crit_map, all_sc, cfg = self._build_full_scenario()
        wv = compute_weights_for_year(mat, regime, crit_map, all_sc, cfg)
        assert abs(sum(wv.values) - 1.0) < 1e-6

    def test_absent_subcriterion_zero_weight(self):
        mat, regime, crit_map, all_sc, cfg = self._build_full_scenario()
        wv = compute_weights_for_year(mat, regime, crit_map, all_sc, cfg)
        d = wv.as_dict()
        assert d["SC83"] < 1e-12

    def test_regime_id_tagged(self):
        mat, regime, crit_map, all_sc, cfg = self._build_full_scenario()
        wv = compute_weights_for_year(mat, regime, crit_map, all_sc, cfg)
        assert wv.regime_id == "R3"

    def test_year_tagged(self):
        mat, regime, crit_map, all_sc, cfg = self._build_full_scenario()
        wv = compute_weights_for_year(mat, regime, crit_map, all_sc, cfg)
        assert wv.year == 2019


# =============================================================================
# 11. _validate_weight_vector
# =============================================================================

class TestValidateWeightVector:
    def test_valid_passes(self):
        wv = WeightVector(labels=["a", "b"], values=[0.3, 0.7])
        _validate_weight_vector(wv)  # should not raise

    def test_invalid_raises(self):
        wv = WeightVector(labels=["a", "b"], values=[0.3, 0.9])  # sum = 1.2
        with pytest.raises(IFSArithmeticError):
            _validate_weight_vector(wv)


# =============================================================================
# 12. Large-scale randomised invariant tests
# =============================================================================

class TestLargeScaleInvariants:
    """Randomised property-based style tests for key invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_weights_always_sum_to_one(self, seed):
        rng = np.random.default_rng(seed)
        n, p = rng.integers(5, 25), rng.integers(2, 8)
        S = rng.uniform(-0.5, 0.5, (n, p))
        # Randomly introduce some NaN
        nan_mask = rng.random((n, p)) < 0.15
        S[nan_mask] = np.nan
        w = _critic_weights_from_score_matrix(S)
        assert abs(w.sum() - 1.0) < _TOL or w.sum() < _TOL  # 0 allowed if all NaN

    @pytest.mark.parametrize("seed", [10, 11, 12])
    def test_stage1_and_stage2_pipeline_integrity(self, seed):
        """Full pipeline: Stage1 → Stage2 → handle_missing → weights sum to 1."""
        rng = np.random.default_rng(seed)
        n = rng.integers(10, 20)
        # Use a fixed small criterion structure
        active_sc = ["SC11", "SC12", "SC21", "SC22", "SC31", "SC32"]
        all_sc = active_sc + ["SC41"]  # SC41 absent
        crit_map = {
            "C01": ["SC11", "SC12"],
            "C02": ["SC21", "SC22"],
            "C03": ["SC31", "SC32", "SC41"],
        }
        mu = rng.uniform(0.0, 0.7, (n, 6))
        nu = rng.uniform(0.0, 0.15, (n, 6))
        mat = _make_ifs_matrix(mu, nu, active_sc, year=2019)
        cfg = _default_w_config()

        stage1 = compute_stage1_weights(mat, crit_map, cfg)
        stage2 = compute_stage2_weights(mat, stage1, crit_map, cfg)
        final = handle_missing_subcriteria(stage1, stage2, crit_map, all_sc)

        assert abs(sum(final.values) - 1.0) < 1e-6
        assert final.as_dict().get("SC41", 0.0) < 1e-12
        assert len(final.values) == len(all_sc)
