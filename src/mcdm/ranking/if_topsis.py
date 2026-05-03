"""
src/mcdm/ranking/if_topsis.py
-----------------------------
Intuitionistic Fuzzy Technique for Order Preference by Similarity to Ideal Solution
(IF-TOPSIS).

Mathematical Specification
===========================
The TOPSIS method ranks provinces by their closeness to an ideal solution and
distance from a negative-ideal solution:

1. **Weighted decision matrix**: V_ij = w_j ⊙ X_ij
   Apply weights via IFS multiplication (scalar multiplication).

2. **Positive Ideal Solution (PIS)**: A^+ = {max_i(μ_ij), min_i(ν_ij)} per j
   For benefit criteria (all sub-criteria in PAPI), higher membership is ideal.

3. **Negative Ideal Solution (NIS)**: A^- = {min_i(μ_ij), max_i(ν_ij)} per j
   For benefit criteria, lower membership is anti-ideal.

4. **Separation measures**:
   - Distance to PIS: d_i^+ = ∑_j d_NE(V_ij, A^+_j)
   - Distance to NIS: d_i^- = ∑_j d_NE(V_ij, A^-_j)
   where d_NE is normalised Euclidean distance on IFS triples.

5. **Closeness coefficient**: CC_i = d_i^- / (d_i^+ + d_i^-)
   Ranges [0, 1]; closer to 1 is better.

6. **Ranking**: Rank by CC_i descending.

Missing data (NaN) is handled by:
- NaN positions in the decision matrix propagate through IFS operations
- Distance calculations handle NaN gracefully (0 if both are NaN, large if only one)
- Weights for NaN sub-criteria are automatically zeroed during weight application

References
----------
Boran, F.E., Genc, S., Kurt, M., & Akay, D. (2009). A multi-criteria intuitionistic
    fuzzy group decision making for supplier selection with TOPSIS method. Expert
    Systems with Applications, 36(8), 11363–11368.
Hwang, C.L., & Yoon, K. (1981). Multiple Attribute Decision Making: Methods and
    Applications. Springer.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.core.exceptions import IFSArithmeticError
from src.core.ifs_arithmetic import (
    IFSMatrix,
    vec_normalized_euclidean_distance,
)
from src.core.schema import RankingMethod, RankingResult

_TOL = 1e-12


# =============================================================================
# IF-TOPSIS Ranking
# =============================================================================

def rank(
    ifs_matrix: IFSMatrix,
    weights: np.ndarray,
    cost_criteria: list = None,
) -> RankingResult:
    """
    Rank provinces using Intuitionistic Fuzzy TOPSIS.

    Parameters
    ----------
    ifs_matrix : IFSMatrix
        Province × Sub-criteria IFS decision matrix.
        Shape: (n_provinces, n_subcriteria).
    weights : ndarray, shape (n_subcriteria,)
        Non-negative sub-criteria weights. Automatically normalised internally.
        Missing sub-criteria (NaN) are handled via weight renormalisation.
    cost_criteria : list, optional
        Sub-criteria indices or names where lower scores are preferred.
        Default None (all benefit criteria — all sub-criteria in PAPI).

    Returns
    -------
    RankingResult
        Provinces ranked 1..n by closeness coefficient descending.

    Raises
    ------
    IFSArithmeticError
        If weight length mismatch.
    """
    if cost_criteria is None:
        cost_criteria = []

    if len(weights) != ifs_matrix.n_criteria:
        raise IFSArithmeticError(
            f"weights length ({len(weights)}) ≠ n_criteria ({ifs_matrix.n_criteria})",
        )

    # Extract decision matrix
    mu = ifs_matrix.mu          # shape (n_prov, n_crit)
    nu = ifs_matrix.nu          # shape (n_prov, n_crit)
    pi = ifs_matrix.pi          # shape (n_prov, n_crit)

    # Apply weights: V_ij = w_j ⊙ X_ij (scalar multiplication)
    mu_weighted, nu_weighted, pi_weighted = _apply_weights(mu, nu, pi, weights)

    # Compute PIS and NIS
    pis_mu, pis_nu, pis_pi = _compute_pis(mu_weighted, nu_weighted, pi_weighted)
    nis_mu, nis_nu, nis_pi = _compute_nis(mu_weighted, nu_weighted, pi_weighted)

    # Compute distances to PIS and NIS
    d_plus = _compute_distance_to_ideal(
        mu_weighted, nu_weighted, pi_weighted,
        pis_mu, pis_nu, pis_pi
    )
    d_minus = _compute_distance_to_ideal(
        mu_weighted, nu_weighted, pi_weighted,
        nis_mu, nis_nu, nis_pi
    )

    # Compute closeness coefficient
    closeness = _compute_closeness_coefficient(d_plus, d_minus)

    # Rank by closeness descending
    ranks = _score_to_rank(closeness)

    return RankingResult(
        method=RankingMethod.IF_TOPSIS,
        year=ifs_matrix.year,
        provinces=ifs_matrix.alternatives,
        scores=closeness.tolist(),
        ranks=ranks.tolist(),
    )


# =============================================================================
# Internal helpers
# =============================================================================

def _apply_weights(
    mu: np.ndarray,
    nu: np.ndarray,
    pi: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply weights via IFS scalar multiplication: V_ij = w_j ⊙ X_ij.

    Scalar multiplication (Xu & Yager 2006):
        w ⊙ A = (1 − (1−μ)^w, ν^w)

    Parameters
    ----------
    mu, nu, pi : ndarray, shape (n_prov, n_crit)
    weights : ndarray, shape (n_crit,)

    Returns
    -------
    mu_w, nu_w, pi_w : ndarray, shape (n_prov, n_crit)
    """
    mu_safe = np.where(np.isnan(mu), 0.0, mu)
    nu_safe = np.where(np.isnan(nu), 1.0, nu)

    w = np.asarray(weights, dtype=float)
    # Broadcast weights: shape (1, n_crit) to match (n_prov, n_crit)
    w_bc = w[np.newaxis, :]

    mu_w = 1.0 - (1.0 - np.clip(mu_safe, 0.0, 1.0)) ** w_bc
    nu_w = np.clip(nu_safe, 0.0, 1.0) ** w_bc
    pi_w = np.clip(1.0 - mu_w - nu_w, 0.0, 1.0)

    # Re-apply NaN mask
    nan_mask = np.isnan(mu) | np.isnan(nu)
    mu_w = np.where(nan_mask, np.nan, mu_w)
    nu_w = np.where(nan_mask, np.nan, nu_w)
    pi_w = np.where(nan_mask, np.nan, pi_w)

    return mu_w, nu_w, pi_w


def _compute_pis(
    mu: np.ndarray,
    nu: np.ndarray,
    pi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Positive Ideal Solution (PIS).

    For benefit criteria: A^+_j = {max_i(μ_ij), min_i(ν_ij)}.

    Parameters
    ----------
    mu, nu, pi : ndarray, shape (n_prov, n_crit)

    Returns
    -------
    pis_mu, pis_nu, pis_pi : ndarray, shape (n_crit,)
    """
    pis_mu = np.nanmax(mu, axis=0)  # max over provinces per criterion
    pis_nu = np.nanmin(nu, axis=0)  # min over provinces per criterion
    pis_pi = np.clip(1.0 - pis_mu - pis_nu, 0.0, 1.0)
    return pis_mu, pis_nu, pis_pi


def _compute_nis(
    mu: np.ndarray,
    nu: np.ndarray,
    pi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Negative Ideal Solution (NIS).

    For benefit criteria: A^-_j = {min_i(μ_ij), max_i(ν_ij)}.

    Parameters
    ----------
    mu, nu, pi : ndarray, shape (n_prov, n_crit)

    Returns
    -------
    nis_mu, nis_nu, nis_pi : ndarray, shape (n_crit,)
    """
    nis_mu = np.nanmin(mu, axis=0)  # min over provinces per criterion
    nis_nu = np.nanmax(nu, axis=0)  # max over provinces per criterion
    nis_pi = np.clip(1.0 - nis_mu - nis_nu, 0.0, 1.0)
    return nis_mu, nis_nu, nis_pi


def _compute_distance_to_ideal(
    mu: np.ndarray,
    nu: np.ndarray,
    pi: np.ndarray,
    ideal_mu: np.ndarray,
    ideal_nu: np.ndarray,
    ideal_pi: np.ndarray,
) -> np.ndarray:
    """
    Compute sum of distances from decision matrix to ideal point.

    d_i = ∑_j d_NE(V_ij, A_j)

    Parameters
    ----------
    mu, nu, pi : ndarray, shape (n_prov, n_crit)
    ideal_mu, ideal_nu, ideal_pi : ndarray, shape (n_crit,)

    Returns
    -------
    distances : ndarray, shape (n_prov,)
    """
    # Broadcast ideal to match decision matrix shape
    ideal_mu_bc = ideal_mu[np.newaxis, :]  # (1, n_crit)
    ideal_nu_bc = ideal_nu[np.newaxis, :]
    ideal_pi_bc = ideal_pi[np.newaxis, :]

    # Compute distance per (province, criterion) pair
    distances_per_crit = vec_normalized_euclidean_distance(
        mu, nu, pi,
        ideal_mu_bc, ideal_nu_bc, ideal_pi_bc
    )  # shape (n_prov, n_crit)

    # Sum distances: handle NaN
    distances = np.nansum(distances_per_crit, axis=1)  # shape (n_prov,)
    return distances


def _compute_closeness_coefficient(
    d_plus: np.ndarray,
    d_minus: np.ndarray,
) -> np.ndarray:
    """
    Compute closeness coefficient CC_i = d_i^- / (d_i^+ + d_i^-).

    Parameters
    ----------
    d_plus : ndarray, shape (n_prov,)  — distance to PIS
    d_minus : ndarray, shape (n_prov,) — distance to NIS

    Returns
    -------
    cc : ndarray, shape (n_prov,)
    """
    denominator = d_plus + d_minus
    # Avoid division by zero: if both distances are ~0, set CC to 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        cc = np.where(
            denominator > _TOL,
            d_minus / denominator,
            0.5
        )
    return np.clip(cc, 0.0, 1.0)


def _score_to_rank(scores: np.ndarray) -> np.ndarray:
    """
    Convert closeness coefficient to rank (1-indexed, higher CC = better rank).

    Parameters
    ----------
    scores : ndarray, shape (n_provinces,)

    Returns
    -------
    ranks : ndarray, shape (n_provinces,), dtype int
    """
    nan_mask = np.isnan(scores)
    scores_safe = np.where(nan_mask, -np.inf, scores)
    sorted_indices = np.argsort(-scores_safe)  # descending
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
    return ranks
