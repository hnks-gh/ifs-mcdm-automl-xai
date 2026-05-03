"""
src/mcdm/weighting/two_level_aggregator.py
------------------------------------------
Two-Level IF-CRITIC weight aggregator.

Orchestrates the full weighting pipeline:

1. **Per-year computation** – for each calendar year, build the IFS matrix,
   run Stage-1 (intra-criterion) and Stage-2 (inter-criterion) CRITIC, then
   combine into a full 29-element sub-criterion weight vector.

2. **Regime blending** – weights computed for a given regime can be averaged
   across years within that regime.  When ``weight_blend_method`` is
   ``"proportional_years"``, each regime's contribution to the cross-regime
   blended weight is proportional to the number of years it covers.

3. **Output** – ``compute_weights_for_all_years`` returns a
   ``dict[int, WeightVector]`` mapping calendar year → per-year final weights
   (over all 29 sub-criteria).

Design notes
------------
* The per-year path is the primary output used by the ranking modules; each
  ranking year gets its own bespoke weight vector reflecting that year's
  province scores.
* Regime-blended weights (``aggregate_regime_weights``) are a secondary
  diagnostic output for temporal stability analysis.
* NaN handling is entirely delegated to :mod:`if_critic`; this module only
  orchestrates calls.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from src.core.exceptions import IFSArithmeticError
from src.core.ifs_arithmetic import IFSMatrix
from src.core.schema import AppConfig, Regime, WeightingConfig, WeightVector
from src.mcdm.weighting.if_critic import (
    compute_stage1_weights,
    compute_stage2_weights,
    handle_missing_subcriteria,
)

logger = logging.getLogger(__name__)

_WEIGHT_TOL: float = 1e-12


# =============================================================================
# Single-year computation
# =============================================================================

def compute_weights_for_year(
    ifs_mat: IFSMatrix,
    regime: Regime,
    criteria_subcriteria_map: Dict[str, List[str]],
    all_subcriteria: List[str],
    weighting_config: WeightingConfig,
) -> WeightVector:
    """
    Compute the full 29-element sub-criterion weight vector for one year.

    Steps
    -----
    1. Stage 1 – CRITIC within each criterion group.
    2. Stage 2 – CRITIC across criterion-level IFS aggregates.
    3. Combine: w_j^final = w_j^(1) * w_k^(2).
    4. Zero-pad to all 29 sub-criteria (absent → 0.0).

    Parameters
    ----------
    ifs_mat : IFSMatrix
        Province × active-sub-criteria IFS matrix for this year.
    regime : Regime
        Regime metadata (used for logging and regime_id tagging).
    criteria_subcriteria_map : dict[str, list[str]]
        Full mapping criterion → list of all its sub-criteria.
    all_subcriteria : list[str]
        Ordered list of all 29 sub-criterion codes.
    weighting_config : WeightingConfig

    Returns
    -------
    WeightVector
        Length 29, active sub-criteria sum to 1.0, absent = 0.0.

    Raises
    ------
    IFSArithmeticError
        If Stage-2 produces no active criteria (degenerate regime).
    """
    year = ifs_mat.year
    logger.info("Computing IF-CRITIC weights for year=%d  regime=%s", year, regime.regime_id)

    # --- Stage 1 ---
    stage1 = compute_stage1_weights(
        ifs_mat=ifs_mat,
        criteria_subcriteria_map=criteria_subcriteria_map,
        config=weighting_config,
    )

    # --- Stage 2 ---
    stage2 = compute_stage2_weights(
        ifs_mat=ifs_mat,
        stage1_weights=stage1,
        criteria_subcriteria_map=criteria_subcriteria_map,
        config=weighting_config,
    )

    # --- Combine into full-length weight vector ---
    final_wv = handle_missing_subcriteria(
        stage1_weights=stage1,
        stage2_weights=stage2,
        criteria_subcriteria_map=criteria_subcriteria_map,
        all_subcriteria=all_subcriteria,
    )
    # Tag regime
    final_wv.regime_id = regime.regime_id
    final_wv.year = year

    _validate_weight_vector(final_wv, context=f"year={year}")
    logger.info(
        "Year=%d weights computed: %d active sub-crit, sum=%.8f",
        year,
        sum(1 for v in final_wv.values if v > _WEIGHT_TOL),
        sum(final_wv.values),
    )
    return final_wv


# =============================================================================
# Regime-level blending
# =============================================================================

def aggregate_regime_weights(
    regime_year_weights: Dict[str, List[WeightVector]],
    regime_year_counts: Dict[str, int],
    all_subcriteria: List[str],
    blend_method: str = "proportional_years",
) -> WeightVector:
    """
    Blend per-year weight vectors into a single cross-regime weight vector.

    For each regime, compute the mean weight vector across its years, then
    blend regimes proportionally by their year counts.

    Parameters
    ----------
    regime_year_weights : dict[str, list[WeightVector]]
        Mapping regime_id → list of per-year WeightVectors for that regime.
    regime_year_counts : dict[str, int]
        Mapping regime_id → total number of years in that regime.
    all_subcriteria : list[str]
        Ordered 29-element list of all sub-criterion codes.
    blend_method : str
        ``"proportional_years"`` (default) or ``"equal_regimes"``.

    Returns
    -------
    WeightVector
        Blended weight vector (length 29) summing to 1.0.

    Notes
    -----
    Active sub-criteria sums to 1.0 after blending.
    Sub-criteria absent in ALL regimes retain weight 0.
    """
    n_sc = len(all_subcriteria)
    sc_index = {sc: i for i, sc in enumerate(all_subcriteria)}

    # Step 1: average within each regime
    regime_mean_weights: Dict[str, np.ndarray] = {}
    for regime_id, wv_list in regime_year_weights.items():
        if not wv_list:
            regime_mean_weights[regime_id] = np.zeros(n_sc)
            continue
        # Stack year vectors into (n_years x 29) array
        arr = np.array([[wv.values[sc_index[sc]] for sc in all_subcriteria]
                        for wv in wv_list], dtype=float)
        regime_mean_weights[regime_id] = arr.mean(axis=0)

    # Step 2: blend regimes
    total_years = sum(regime_year_counts.values())
    if total_years == 0:
        raise IFSArithmeticError("aggregate_regime_weights: total year count is zero")

    blended = np.zeros(n_sc, dtype=float)
    for regime_id, mean_w in regime_mean_weights.items():
        if blend_method == "proportional_years":
            weight = regime_year_counts.get(regime_id, 0) / total_years
        elif blend_method == "equal_regimes":
            weight = 1.0 / len(regime_mean_weights)
        else:
            raise IFSArithmeticError(
                f"Unknown blend_method: {blend_method}",
                context={"blend_method": blend_method},
            )
        blended += weight * mean_w

    # Re-normalise over active sub-criteria
    w_sum = float(blended.sum())
    if w_sum > _WEIGHT_TOL:
        blended = blended / w_sum

    return WeightVector(
        labels=list(all_subcriteria),
        values=blended.tolist(),
        year=None,
        regime_id="blended",
        stage=None,
    )


# =============================================================================
# Full panel computation (main entry point)
# =============================================================================

def compute_weights_for_all_years(
    ifs_panel: Dict[int, IFSMatrix],
    regimes: Dict[str, Regime],
    config: AppConfig,
) -> Dict[int, WeightVector]:
    """
    Compute two-level IF-CRITIC weight vectors for every year in the panel.

    Parameters
    ----------
    ifs_panel : dict[int, IFSMatrix]
        Output of :func:`src.core.preprocessor.convert_panel_to_ifs`.
    regimes : dict[str, Regime]
        Regime metadata from :func:`src.core.data_loader.detect_regimes`.
    config : AppConfig
        Full application configuration.

    Returns
    -------
    dict[int, WeightVector]
        Mapping calendar year → full 29-element WeightVector.
        Active sub-criterion weights sum to 1.0 for each year.

    Raises
    ------
    IFSArithmeticError
        On any degenerate regime (no active criteria).
    """
    from src.core.data_loader import get_regime_for_year

    criteria_map: Dict[str, List[str]] = config.data.criteria_subcriteria_map
    all_sc: List[str] = config.data.all_subcriteria
    w_cfg: WeightingConfig = config.mcdm.weighting

    year_weights: Dict[int, WeightVector] = {}

    for year in sorted(ifs_panel.keys()):
        ifs_mat = ifs_panel[year]
        try:
            regime = get_regime_for_year(year, regimes)
        except Exception as exc:
            logger.error("Cannot find regime for year %d: %s", year, exc)
            raise

        wv = compute_weights_for_year(
            ifs_mat=ifs_mat,
            regime=regime,
            criteria_subcriteria_map=criteria_map,
            all_subcriteria=all_sc,
            weighting_config=w_cfg,
        )
        year_weights[year] = wv

    logger.info(
        "compute_weights_for_all_years: weights computed for %d years: %s",
        len(year_weights),
        sorted(year_weights.keys()),
    )
    return year_weights


# =============================================================================
# Compute final sub-criteria weights  (Stage1 x Stage2 combined)
# =============================================================================

def compute_final_subcriteria_weights(
    stage1: Dict[str, WeightVector],
    stage2: WeightVector,
    criteria_subcriteria_map: Dict[str, List[str]],
    all_subcriteria: List[str],
) -> WeightVector:
    """
    Combine Stage-1 and Stage-2 weights into the 29-element final weight vector.

    Thin public wrapper around :func:`if_critic.handle_missing_subcriteria`
    for use by external callers who have already computed Stage-1 and Stage-2.

    Parameters
    ----------
    stage1 : dict[str, WeightVector]
    stage2 : WeightVector
    criteria_subcriteria_map : dict[str, list[str]]
    all_subcriteria : list[str]

    Returns
    -------
    WeightVector
        Length 29, active weights sum to 1.0.
    """
    from src.mcdm.weighting.if_critic import handle_missing_subcriteria
    return handle_missing_subcriteria(
        stage1_weights=stage1,
        stage2_weights=stage2,
        criteria_subcriteria_map=criteria_subcriteria_map,
        all_subcriteria=all_subcriteria,
    )


# =============================================================================
# Internal validation helper
# =============================================================================

def _validate_weight_vector(wv: WeightVector, context: str = "") -> None:
    """
    Assert that active weights sum to 1.0 within floating-point tolerance.

    Raises
    ------
    IFSArithmeticError
        If the sum deviates from 1.0 by more than 1e-6.
    """
    w_sum = sum(wv.values)
    if abs(w_sum - 1.0) > 1e-6:
        raise IFSArithmeticError(
            f"WeightVector sum={w_sum:.8f} ≠ 1.0  [{context}]",
            context={"sum": w_sum, "context": context},
        )
