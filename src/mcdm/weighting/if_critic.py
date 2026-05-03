"""
src/mcdm/weighting/if_critic.py
--------------------------------
Intuitionistic Fuzzy CRITIC (IF-CRITIC) weighting engine.

Mathematical Foundation
-----------------------
CRITIC (CRiteria Importance Through Intercriteria Correlation) was introduced
by Diakoulaki et al. (1995) and extended here to Intuitionistic Fuzzy Sets by
operating on the score function S(x) = mu - nu, which maps each IFS number to
a crisp value in [-1, 1].

**Stage 1 – Intra-criterion sub-criteria weights**

For each criterion group Ck containing p active sub-criteria, and an
(n x p) matrix of IFS observations X:

1. Score matrix: S_ij = mu_ij - nu_ij  (shape n x p, NaN preserved)
2. Column standard deviation: sigma_j = std(S_ij, ddof=1) over valid rows
3. IFS-adapted Pearson correlation: r_jl = corr(S_j, S_l) on pairwise valid obs
4. CRITIC information: C_j = sigma_j * sum_{l=1}^{p} (1 - r_jl)
5. Stage-1 weight: w_j^(1) = C_j / sum_l C_l

If a sub-criterion has zero variance (all values identical), it contributes
zero to C_j (no information), so it receives zero weight.

**Stage 2 – Inter-criterion (global criteria) weights**

1. Aggregate each province's sub-criteria into a criterion-level IFS number
   using IFS Weighted Arithmetic Mean (IFS-WAM) with Stage-1 weights.
2. Repeat CRITIC on the resulting (n x K) matrix of criterion-level IFS
   aggregates.

**Final combined weight**

    w_j^final = w_j^(1) * w_k^(2)

summing over all active sub-criteria across all criteria gives 1.0.

NaN Handling
------------
- Missing province values (Type-3 partial NaN) are excluded pairwise from
  correlation and listwise from std computation.
- If a sub-criterion column is entirely NaN (absent in this regime), it is
  assigned zero weight and excluded from normalization.
- If a criterion group has only one active sub-criterion, its Stage-1 weight
  is trivially 1.0 (no correlation to compute).
- If all criteria aggregates are identical (zero variance at Stage 2), equal
  weights are assigned.

References
----------
Diakoulaki, D., Mavrotas, G., Papayannakis, L. (1995). Determining objective
    weights in multiple criteria problems: The CRITIC method. Comput. Oper.
    Res. 22(7), 763-770.
Xu, Z., Yager, R.R. (2006). Some geometric aggregation operators based on
    intuitionistic fuzzy sets. Int. J. Gen. Syst. 35(4), 417-433.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.exceptions import IFSArithmeticError
from src.core.ifs_arithmetic import IFSMatrix
from src.core.schema import Regime, WeightingConfig, WeightVector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_MIN_VALID_OBS: int = 2          # minimum non-NaN rows needed for std/corr
_WEIGHT_TOL: float = 1e-12       # tolerance for weight sum checks
_CORR_CLIP: float = 1.0          # correlation values clipped to [-1, 1]


# =============================================================================
# Low-level CRITIC kernel (operates on raw 2-D float arrays)
# =============================================================================

def _compute_std_nanaware(S: np.ndarray) -> np.ndarray:
    """
    Compute column-wise sample standard deviation, ignoring NaN.

    Parameters
    ----------
    S : ndarray, shape (n, p)
        Score-function matrix.

    Returns
    -------
    sigma : ndarray, shape (p,)
        Column std. Zero for columns with < 2 valid observations.
    """
    p = S.shape[1]
    sigma = np.zeros(p, dtype=float)
    for j in range(p):
        col = S[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) >= _MIN_VALID_OBS:
            sigma[j] = float(np.std(valid, ddof=1))
    return sigma


def _compute_corr_matrix_nanaware(S: np.ndarray) -> np.ndarray:
    """
    Compute the (p x p) Pearson correlation matrix with pairwise NaN deletion.

    For each pair (j, l), only rows where both columns are non-NaN are used.
    If fewer than 2 shared valid observations exist, r_jl is set to 0.0
    (no correlation assumed → conservative: column contributes full (1 - 0) = 1).

    Parameters
    ----------
    S : ndarray, shape (n, p)

    Returns
    -------
    R : ndarray, shape (p, p)
        Symmetric correlation matrix with diagonal 1.0.
    """
    n, p = S.shape
    R = np.eye(p, dtype=float)

    for j in range(p):
        for l in range(j + 1, p):
            col_j = S[:, j]
            col_l = S[:, l]
            both_valid = ~np.isnan(col_j) & ~np.isnan(col_l)
            n_valid = int(both_valid.sum())

            if n_valid < _MIN_VALID_OBS:
                r = 0.0
            else:
                v_j = col_j[both_valid]
                v_l = col_l[both_valid]
                # Pearson r; protect against zero-variance columns
                std_j = float(np.std(v_j, ddof=1))
                std_l = float(np.std(v_l, ddof=1))
                if std_j < _WEIGHT_TOL or std_l < _WEIGHT_TOL:
                    r = 0.0
                else:
                    r = float(np.corrcoef(v_j, v_l)[0, 1])
                    r = float(np.clip(r, -_CORR_CLIP, _CORR_CLIP))

            R[j, l] = r
            R[l, j] = r

    return R


def _critic_weights_from_score_matrix(
    S: np.ndarray,
    min_variance_threshold: float = 1e-9,
) -> np.ndarray:
    """
    Core CRITIC computation from a score-function matrix.

    Parameters
    ----------
    S : ndarray, shape (n, p)
        Score values S = mu - nu; NaN allowed.
    min_variance_threshold : float
        Columns whose variance < threshold are treated as zero-variance.

    Returns
    -------
    weights : ndarray, shape (p,)
        Non-negative weights summing to 1.0.
        If all C_j == 0, returns equal weights over non-entirely-NaN columns.
    """
    n, p = S.shape

    if p == 0:
        return np.array([], dtype=float)

    # Identify entirely-NaN columns (structurally absent)
    entirely_nan = np.all(np.isnan(S), axis=0)   # shape (p,)

    if p == 1:
        # Trivial case: single sub-criterion gets weight 1 (if not entirely NaN)
        return np.array([0.0 if entirely_nan[0] else 1.0])

    # Standard deviations
    sigma = _compute_std_nanaware(S)

    # Zero out variance-below-threshold columns
    sigma[sigma ** 2 < min_variance_threshold] = 0.0

    # Zero out entirely-NaN columns
    sigma[entirely_nan] = 0.0

    # Correlation matrix
    R = _compute_corr_matrix_nanaware(S)

    # CRITIC information measure: C_j = sigma_j * sum_l (1 - r_jl)
    # sum_l (1 - r_jl) = p - sum_l r_jl
    conflict = np.sum(1.0 - R, axis=1)   # shape (p,)
    C = sigma * conflict                   # shape (p,)

    # Zero out entirely-NaN / zero-variance columns
    C[entirely_nan] = 0.0
    C[sigma == 0.0] = 0.0

    C_sum = float(C.sum())
    if C_sum < _WEIGHT_TOL:
        # Degenerate: assign equal weight to active (non-entirely-NaN) columns
        n_active = int((~entirely_nan).sum())
        weights = np.where(entirely_nan, 0.0, 1.0 / n_active if n_active > 0 else 0.0)
    else:
        weights = C / C_sum

    return weights


# =============================================================================
# Stage 1 – intra-criterion CRITIC (sub-criteria within each criterion)
# =============================================================================

def compute_critic_weights(
    score_matrix: np.ndarray,
    labels: List[str],
    min_variance_threshold: float = 1e-9,
) -> np.ndarray:
    """
    Compute CRITIC weights from a score-function matrix.

    Public thin wrapper around ``_critic_weights_from_score_matrix`` that adds
    logging and label validation.

    Parameters
    ----------
    score_matrix : ndarray, shape (n_provinces, p)
        S = mu - nu values for the sub-criteria being weighted.
        NaN indicates missing province-year observations.
    labels : list[str]
        Names of the p sub-criteria (for logging).
    min_variance_threshold : float
        Columns with variance below this are treated as zero-information.

    Returns
    -------
    weights : ndarray, shape (p,)
        Non-negative, summing to 1.0.
    """
    p = score_matrix.shape[1] if score_matrix.ndim == 2 else 0
    if len(labels) != p:
        raise IFSArithmeticError(
            f"labels length ({len(labels)}) != score_matrix columns ({p})",
            context={"labels": labels, "p": p},
        )

    weights = _critic_weights_from_score_matrix(score_matrix, min_variance_threshold)
    logger.debug(
        "compute_critic_weights: p=%d  weights=%s",
        p,
        {lbl: f"{w:.4f}" for lbl, w in zip(labels, weights)},
    )
    return weights


def compute_stage1_weights(
    ifs_mat: IFSMatrix,
    criteria_subcriteria_map: Dict[str, List[str]],
    config: WeightingConfig,
) -> Dict[str, WeightVector]:
    """
    Stage 1: Compute intra-criterion CRITIC weights for all criterion groups.

    For each criterion Ck, extracts the sub-matrix of active sub-criteria,
    computes score function S = mu - nu, then runs CRITIC.

    Parameters
    ----------
    ifs_mat : IFSMatrix
        Full year-specific IFS matrix (n_provinces x n_active_subcriteria).
    criteria_subcriteria_map : dict[str, list[str]]
        Mapping criterion_code -> ordered list of all sub-criteria in that
        criterion (including potentially absent ones).
    config : WeightingConfig
        Weighting configuration (correlation_method, min_variance_threshold).

    Returns
    -------
    dict[str, WeightVector]
        Keyed by criterion code.  Each WeightVector covers the sub-criteria
        **present in the IFSMatrix** for that criterion (absent sub-criteria
        will have zero weights added later by the aggregator).
    """
    S_full = ifs_mat.mu - ifs_mat.nu   # shape (n_alts, n_active_crit)

    stage1: Dict[str, WeightVector] = {}

    for criterion_code, all_sc_in_criterion in criteria_subcriteria_map.items():
        # Identify which sub-criteria of this criterion are present in the matrix
        sc_present = [sc for sc in all_sc_in_criterion if sc in ifs_mat.criteria]

        if not sc_present:
            # This criterion has no active sub-criteria in this year/regime
            logger.debug(
                "Stage1: criterion %s has no active sub-criteria in year %d – skipped",
                criterion_code, ifs_mat.year,
            )
            stage1[criterion_code] = WeightVector(
                labels=[],
                values=[],
                year=ifs_mat.year,
                regime_id=None,
                stage=1,
            )
            continue

        # Extract columns for this criterion group
        col_indices = [ifs_mat.criteria.index(sc) for sc in sc_present]
        S_group = S_full[:, col_indices]   # shape (n_alts, len(sc_present))

        weights_arr = compute_critic_weights(
            score_matrix=S_group,
            labels=sc_present,
            min_variance_threshold=config.min_variance_threshold,
        )

        stage1[criterion_code] = WeightVector(
            labels=sc_present,
            values=weights_arr.tolist(),
            year=ifs_mat.year,
            regime_id=None,
            stage=1,
        )
        logger.debug(
            "Stage1 year=%d criterion=%s: %d sub-crit active, weights sum=%.6f",
            ifs_mat.year, criterion_code,
            len(sc_present), float(np.sum(weights_arr)),
        )

    return stage1


# =============================================================================
# IFS-WAM aggregation helper (for Stage 2 criterion-level aggregates)
# =============================================================================

def _ifs_wam_aggregate(
    mu: np.ndarray,
    nu: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised IFS Weighted Arithmetic Mean over columns.

    IFS-WAM(A_1,...,A_p; w_1,...,w_p):
        mu_agg   = 1 - prod_j (1 - mu_j)^w_j
        nu_agg   = prod_j nu_j^w_j
        pi_agg   = 1 - mu_agg - nu_agg

    Formula from Xu & Yager (2006).  Only non-NaN, positive-weight columns
    participate.  If all columns are NaN for a province row, the aggregate
    is NaN.

    Parameters
    ----------
    mu : ndarray, shape (n, p)
    nu : ndarray, shape (n, p)
    weights : ndarray, shape (p,)
        Non-negative, should sum to 1 over active (non-zero) columns.

    Returns
    -------
    mu_agg, nu_agg, pi_agg : ndarray, shape (n,)
    """
    n, p = mu.shape
    mu_agg = np.full(n, np.nan, dtype=float)
    nu_agg = np.full(n, np.nan, dtype=float)

    for i in range(n):
        # Identify valid (non-NaN) columns with positive weight
        mu_row = mu[i, :]
        nu_row = nu[i, :]
        valid = ~np.isnan(mu_row) & ~np.isnan(nu_row) & (weights > _WEIGHT_TOL)

        if not valid.any():
            continue  # all NaN or zero weight → aggregate stays NaN

        w_valid = weights[valid]
        # Renormalize weights over valid columns so they still sum to 1
        w_sum = float(w_valid.sum())
        if w_sum < _WEIGHT_TOL:
            continue
        w_norm = w_valid / w_sum

        mu_valid = mu_row[valid]
        nu_valid = nu_row[valid]

        # Clip to [0,1] for numerical safety before power operations
        mu_valid = np.clip(mu_valid, 0.0, 1.0)
        nu_valid = np.clip(nu_valid, 0.0, 1.0)

        # IFS-WAM formulas
        prod_one_minus_mu = float(np.prod((1.0 - mu_valid) ** w_norm))
        prod_nu = float(np.prod(nu_valid ** w_norm))

        mu_agg[i] = max(0.0, 1.0 - prod_one_minus_mu)
        nu_agg[i] = max(0.0, prod_nu)

        # Ensure mu + nu <= 1 (numerical safety)
        if mu_agg[i] + nu_agg[i] > 1.0:
            excess = mu_agg[i] + nu_agg[i] - 1.0
            # Proportionally shrink
            total = mu_agg[i] + nu_agg[i]
            mu_agg[i] = mu_agg[i] / total
            nu_agg[i] = nu_agg[i] / total

    pi_agg = np.where(
        np.isnan(mu_agg),
        np.nan,
        np.clip(1.0 - mu_agg - nu_agg, 0.0, 1.0),
    )
    return mu_agg, nu_agg, pi_agg


# =============================================================================
# Stage 2 – inter-criterion CRITIC (criteria-level global weights)
# =============================================================================

def compute_stage2_weights(
    ifs_mat: IFSMatrix,
    stage1_weights: Dict[str, WeightVector],
    criteria_subcriteria_map: Dict[str, List[str]],
    config: WeightingConfig,
) -> WeightVector:
    """
    Stage 2: Compute inter-criterion CRITIC weights from criterion aggregates.

    For each active criterion Ck (having at least one active sub-criterion in
    this year), aggregate provinces' sub-criteria IFS values using Stage-1
    weights → criterion-level IFS column.  Then run CRITIC on the resulting
    (n x K_active) matrix.

    Parameters
    ----------
    ifs_mat : IFSMatrix
        Full year-specific IFS matrix.
    stage1_weights : dict[str, WeightVector]
        Output of :func:`compute_stage1_weights`.
    criteria_subcriteria_map : dict[str, list[str]]
        Full mapping (all 8 criteria).
    config : WeightingConfig

    Returns
    -------
    WeightVector
        Global criteria weights (length = number of active criteria in year).
        Labels are criterion codes (e.g. "C01", "C02", ...).
    """
    n = ifs_mat.n_alternatives
    criterion_codes_active: List[str] = []
    mu_criteria_list: List[np.ndarray] = []
    nu_criteria_list: List[np.ndarray] = []

    for criterion_code in sorted(criteria_subcriteria_map.keys()):
        wv = stage1_weights.get(criterion_code)
        if wv is None or len(wv.labels) == 0:
            # No active sub-criteria for this criterion in this year/regime
            logger.debug(
                "Stage2 year=%d: criterion %s has no active sub-criteria – excluded",
                ifs_mat.year, criterion_code,
            )
            continue

        # Gather sub-criteria columns in the order declared by Stage-1 weights
        sc_labels = wv.labels
        sc_weights = np.array(wv.values, dtype=float)

        col_indices = [ifs_mat.criteria.index(sc) for sc in sc_labels]
        mu_block = ifs_mat.mu[:, col_indices]   # (n, p_k)
        nu_block = ifs_mat.nu[:, col_indices]   # (n, p_k)

        # IFS-WAM → criterion-level aggregate per province
        mu_agg, nu_agg, _ = _ifs_wam_aggregate(mu_block, nu_block, sc_weights)

        criterion_codes_active.append(criterion_code)
        mu_criteria_list.append(mu_agg)
        nu_criteria_list.append(nu_agg)

    K = len(criterion_codes_active)
    if K == 0:
        raise IFSArithmeticError(
            "Stage 2: no active criteria found for this year/regime",
            context={"year": ifs_mat.year},
        )

    # Build (n x K) score matrix for CRITIC
    mu_mat = np.stack(mu_criteria_list, axis=1)   # (n, K)
    nu_mat = np.stack(nu_criteria_list, axis=1)   # (n, K)
    S_mat = mu_mat - nu_mat                        # (n, K)

    weights_arr = compute_critic_weights(
        score_matrix=S_mat,
        labels=criterion_codes_active,
        min_variance_threshold=config.min_variance_threshold,
    )

    logger.debug(
        "Stage2 year=%d: %d active criteria, weights=%s, sum=%.6f",
        ifs_mat.year, K,
        {c: f"{w:.4f}" for c, w in zip(criterion_codes_active, weights_arr)},
        float(weights_arr.sum()),
    )

    return WeightVector(
        labels=criterion_codes_active,
        values=weights_arr.tolist(),
        year=ifs_mat.year,
        regime_id=None,
        stage=2,
    )


# =============================================================================
# Handle missing sub-criteria (zero-pad to full 29-element space)
# =============================================================================

def handle_missing_subcriteria(
    stage1_weights: Dict[str, WeightVector],
    stage2_weights: WeightVector,
    criteria_subcriteria_map: Dict[str, List[str]],
    all_subcriteria: List[str],
) -> WeightVector:
    """
    Compute final combined sub-criterion weights over the full 29-element space.

    For each sub-criterion j in criterion k:
        w_j^final = w_j^(1) * w_k^(2)

    Absent sub-criteria (not active in this year) receive weight 0.0.
    The returned vector is over ``all_subcriteria`` in the given order.

    Parameters
    ----------
    stage1_weights : dict[str, WeightVector]
        Intra-criterion weights from Stage 1.
    stage2_weights : WeightVector
        Inter-criterion weights from Stage 2.
    criteria_subcriteria_map : dict[str, list[str]]
        Full mapping of all criteria → all sub-criteria.
    all_subcriteria : list[str]
        Ordered list of all 29 sub-criterion codes.

    Returns
    -------
    WeightVector
        Length = len(all_subcriteria).  Active sub-criteria receive combined
        weight; absent receive 0.0.  Active weights sum to 1.0.
    """
    # Build lookup: criterion_code -> stage2 weight
    s2_dict = dict(zip(stage2_weights.labels, stage2_weights.values))

    # Build lookup: sub_criterion_code -> final weight
    final_weights: Dict[str, float] = {sc: 0.0 for sc in all_subcriteria}

    for criterion_code, sc_list in criteria_subcriteria_map.items():
        w_k = s2_dict.get(criterion_code, 0.0)   # Stage-2 weight for this criterion
        if w_k < _WEIGHT_TOL:
            # Criterion not active or zero Stage-2 weight → sub-criteria get 0
            continue

        wv = stage1_weights.get(criterion_code)
        if wv is None or len(wv.labels) == 0:
            continue

        for sc, w_j_within in zip(wv.labels, wv.values):
            if sc in final_weights:
                final_weights[sc] = w_j_within * w_k

    values = [final_weights[sc] for sc in all_subcriteria]
    w_sum = sum(values)

    # Normalise to correct floating-point drift
    if w_sum > _WEIGHT_TOL:
        values = [v / w_sum for v in values]

    logger.debug(
        "handle_missing_subcriteria: total active sub-crit with weight>0: %d, sum=%.8f",
        sum(1 for v in values if v > _WEIGHT_TOL),
        sum(values),
    )

    return WeightVector(
        labels=list(all_subcriteria),
        values=values,
        year=stage2_weights.year,
        regime_id=stage2_weights.regime_id,
        stage=None,  # combined final weight
    )
