"""
src/mcdm/ranking/if_waspas.py
-----------------------------
Intuitionistic Fuzzy Weighted Aggregated Sum Product Assessment (IF-WASPAS).

Mathematical Specification
===========================
The WASPAS method combines two aggregation strategies via a balance parameter λ ∈ [0, 1]:

1. **Weighted Sum Model (WSM) component**: Q_i^(1) = IF-WAM({x_ij}, {w_j})
   Intuitionistic Fuzzy Weighted Arithmetic Mean over all sub-criteria.

2. **Weighted Product Model (WPM) component**: Q_i^(2) = IF-WGM({x_ij}, {w_j})
   Intuitionistic Fuzzy Weighted Geometric Mean over all sub-criteria.

3. **Blended final score**: Q_i = λ ⊕ Q_i^(1) ⊕ ((1−λ) ⊗ Q_i^(2))
   Uses IFS addition (⊕) and scalar multiplication.

4. **Ranking**: Rank provinces by S(Q_i) = μ(Q_i) − ν(Q_i) in descending order.
   Higher score → better rank.

Missing data (NaN) is handled via weight re-normalisation: weights are internally
normalised across non-NaN sub-criteria only, so missing data automatically receives
zero weight contribution.

References
----------
Chakraborty, S. (2011). Applications of the MOORA method for decision making in
    manufacturing environment. International Journal of Advanced Manufacturing
    Technology, 54(5-8), 771-784.
Paradowski, B. (2016). Intuitionistic fuzzy WASPAS method.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.core.exceptions import IFSArithmeticError
from src.core.ifs_arithmetic import IFSMatrix, vec_wam, vec_wgm
from src.core.schema import RankingMethod, RankingResult


# =============================================================================
# IF-WASPAS Ranking
# =============================================================================

def rank(
    ifs_matrix: IFSMatrix,
    weights: np.ndarray,
    lambda_param: float = 0.5,
) -> RankingResult:
    """
    Rank provinces using Intuitionistic Fuzzy WASPAS.

    Parameters
    ----------
    ifs_matrix : IFSMatrix
        Province × Sub-criteria IFS decision matrix.
        Shape: (n_provinces, n_subcriteria).
    weights : ndarray, shape (n_subcriteria,)
        Non-negative sub-criteria weights.  Automatically normalised internally.
        Missing sub-criteria (NaN in IFS matrix) are automatically excluded.
    lambda_param : float, optional
        Balance parameter λ ∈ [0, 1].
        - λ = 0: pure WPM (geometric mean)
        - λ = 0.5: balanced (default)
        - λ = 1: pure WSM (arithmetic mean)

    Returns
    -------
    RankingResult
        Provinces ranked 1..n by composite score descending.
        Higher score → better rank.

    Raises
    ------
    IFSArithmeticError
        If lambda_param not in [0, 1] or weight length mismatch.
    """
    if not (0.0 <= lambda_param <= 1.0):
        raise IFSArithmeticError(
            f"lambda_param must be in [0, 1], got {lambda_param}",
            context={"lambda": lambda_param},
        )
    if len(weights) != ifs_matrix.n_criteria:
        raise IFSArithmeticError(
            f"weights length ({len(weights)}) ≠ n_criteria ({ifs_matrix.n_criteria})",
        )

    # Extract decision matrix
    mu = ifs_matrix.mu         # shape (n_prov, n_crit)
    nu = ifs_matrix.nu         # shape (n_prov, n_crit)

    # Compute WSM: Q_i^(1) = IFS-WAM
    mu_wsm, nu_wsm, _ = vec_wam(mu, nu, weights)

    # Compute WPM: Q_i^(2) = IFS-WGM
    mu_wpm, nu_wpm, _ = vec_wgm(mu, nu, weights)

    # Blend: Q_i = λ * Q_i^(1) ⊕ (1-λ) * Q_i^(2)
    # This requires IFS scalar multiplication and addition (vectorised)
    mu_final, nu_final, _ = _blend_wsm_wpm(
        mu_wsm, nu_wsm,
        mu_wpm, nu_wpm,
        lambda_param
    )

    # Compute scores: S(Q_i) = μ - ν
    scores = mu_final - nu_final

    # Rank provinces by score descending (higher score = rank 1)
    ranks = _score_to_rank(scores)

    return RankingResult(
        method=RankingMethod.IF_WASPAS,
        year=ifs_matrix.year,
        provinces=ifs_matrix.alternatives,
        scores=scores.tolist(),
        ranks=ranks.tolist(),
    )


# =============================================================================
# Internal helpers
# =============================================================================

def _blend_wsm_wpm(
    mu_wsm: np.ndarray,
    nu_wsm: np.ndarray,
    mu_wpm: np.ndarray,
    nu_wpm: np.ndarray,
    lambda_param: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Blend WSM and WPM components: Q = λ * Q^(1) ⊕ (1-λ) * Q^(2).

    Implements:
        Q = λA ⊕ (1-λ)B
        where A = (μ_WSM, ν_WSM), B = (μ_WPM, ν_WPM)

    Operations:
        λA = (1−(1−μ_WSM)^λ, ν_WSM^λ)
        (1−λ)B = (1−(1−μ_WPM)^(1−λ), ν_WPM^(1−λ))
        λA ⊕ (1−λ)B = ((1−(1−μ_A)^...)(1−(1−μ_B)^...), ...)

    Parameters
    ----------
    mu_wsm, nu_wsm : ndarray, shape (n_provinces,)
    mu_wpm, nu_wpm : ndarray, shape (n_provinces,)
    lambda_param : float ∈ [0, 1]

    Returns
    -------
    mu_final, nu_final, pi_final : ndarray, shape (n_provinces,)
    """
    _TOL = 1e-9

    # Scalar multiplication: λA and (1-λ)B
    # λA = (1−(1−μ_WSM)^λ, ν_WSM^λ)
    mu_lam_wsm = 1.0 - (1.0 - np.clip(mu_wsm, 0.0, 1.0)) ** lambda_param
    nu_lam_wsm = np.clip(nu_wsm, 0.0, 1.0) ** lambda_param

    # (1-λ)B = (1−(1−μ_WPM)^(1−λ), ν_WPM^(1−λ))
    one_minus_lam = 1.0 - lambda_param
    if abs(one_minus_lam) > _TOL:  # avoid 0^0 if lambda_param == 1
        mu_1mlam_wpm = 1.0 - (1.0 - np.clip(mu_wpm, 0.0, 1.0)) ** one_minus_lam
        nu_1mlam_wpm = np.clip(nu_wpm, 0.0, 1.0) ** one_minus_lam
    else:
        # If lambda_param == 1, then (1-λ)B = 0B = (0, 1) ~ neutral for union
        mu_1mlam_wpm = np.zeros_like(mu_wpm)
        nu_1mlam_wpm = np.ones_like(nu_wpm)

    # IFS addition: (μ_a + μ_b − μ_a*μ_b, ν_a*ν_b)
    mu_final = mu_lam_wsm + mu_1mlam_wpm - mu_lam_wsm * mu_1mlam_wpm
    nu_final = nu_lam_wsm * nu_1mlam_wpm

    # Clip to valid range and ensure π ≥ 0
    mu_final = np.clip(mu_final, 0.0, 1.0)
    nu_final = np.clip(nu_final, 0.0, 1.0)
    # If mu + nu > 1, clip nu to restore constraint
    nu_final = np.minimum(nu_final, 1.0 - mu_final)

    pi_final = np.clip(1.0 - mu_final - nu_final, 0.0, 1.0)

    return mu_final, nu_final, pi_final


def _score_to_rank(scores: np.ndarray) -> np.ndarray:
    """
    Convert score array to rank array (1-indexed, higher score = better rank = lower number).

    Parameters
    ----------
    scores : ndarray, shape (n_provinces,)

    Returns
    -------
    ranks : ndarray, shape (n_provinces,), dtype int
        Rank for each province (1 = best).
    """
    # Handle NaN by assigning worst rank
    nan_mask = np.isnan(scores)
    scores_safe = np.where(nan_mask, -np.inf, scores)
    # Argsort: lowest index gets rank 1
    sorted_indices = np.argsort(-scores_safe)  # descending
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
    return ranks
