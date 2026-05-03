"""
src/mcdm/ranking/if_promethee2.py
---------------------------------
Intuitionistic Fuzzy PROMETHEE II (Preference Ranking Organization Method
for Enrichment Evaluation).

Mathematical Specification
===========================
PROMETHEE II ranks provinces based on pairwise preference comparisons across
all criteria:

1. **Preference functions** P_j(i, k) quantify how much province i is preferred
   over province k on criterion j. Uses Gaussian preference function:

   P_j(i, k) = 0                           if d ≤ 0
              = 1 − exp(−d²/(2p²))         if d > 0
   where d = S(X_ij) − S(X_kj) and S(A) = μ − ν.

2. **Weighted preference** π(i, k) = ∑_j w_j · P_j(i, k)
   Aggregates across all criteria.

3. **Positive outranking flow** φ⁺(i) = (1/(n−1)) · ∑_k π(i, k)
   How strongly i outranks others.

4. **Negative outranking flow** φ⁻(i) = (1/(n−1)) · ∑_k π(k, i)
   How strongly others outrank i.

5. **Net outranking flow** φ(i) = φ⁺(i) − φ⁻(i)
   Net preference; used for final ranking.

6. **Ranking**: Rank by φ(i) descending (higher net flow = better rank).

Missing data (NaN) is handled by:
- Pairwise comparisons involving NaN skip those criteria (weight renormalisation)
- Provinces with all NaN sub-criteria receive neutral flow values

References
----------
Brans, J.P., & Vincke, Ph. (1985). A preference ranking organisation method.
    Management Science, 31(6), 647–656.
Brans, J.P., Vincke, Ph., & Mareschal, B. (1986). How to select and how to rank
    projects: The PROMETHEE method. European Journal of Operational Research,
    24(2), 228–238.
Liang, W., Zhao, G., & Wu, H. (2019). Evaluating innovation capability of high-tech
    enterprises based on multi-criteria decision making methods with intuitionistic
    fuzzy information. Journal of Intelligent & Fuzzy Systems, 36(4), 3001–3010.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.core.exceptions import IFSArithmeticError
from src.core.ifs_arithmetic import IFSMatrix
from src.core.schema import RankingMethod, RankingResult

_TOL = 1e-12


# =============================================================================
# IF-PROMETHEE II Ranking
# =============================================================================

def rank(
    ifs_matrix: IFSMatrix,
    weights: np.ndarray,
    p_parameter: float = 0.1,
    preference_function: str = "gaussian",
) -> RankingResult:
    """
    Rank provinces using Intuitionistic Fuzzy PROMETHEE II.

    Parameters
    ----------
    ifs_matrix : IFSMatrix
        Province × Sub-criteria IFS decision matrix.
        Shape: (n_provinces, n_subcriteria).
    weights : ndarray, shape (n_subcriteria,)
        Non-negative sub-criteria weights. Automatically normalised internally.
        Missing sub-criteria (NaN) are handled via weight renormalisation.
    p_parameter : float, optional
        Indifference parameter p for Gaussian preference function.
        Default 0.1. Must be > 0.
    preference_function : str, optional
        Preference function type. Only "gaussian" supported.
        Default "gaussian".

    Returns
    -------
    RankingResult
        Provinces ranked 1..n by net outranking flow descending.

    Raises
    ------
    IFSArithmeticError
        If parameters invalid.
    """
    if p_parameter <= 0:
        raise IFSArithmeticError(
            f"p_parameter must be > 0, got {p_parameter}",
            context={"p": p_parameter},
        )
    if preference_function != "gaussian":
        raise IFSArithmeticError(
            f"Unknown preference function '{preference_function}'",
            context={"pref_fn": preference_function},
        )
    if len(weights) != ifs_matrix.n_criteria:
        raise IFSArithmeticError(
            f"weights length ({len(weights)}) ≠ n_criteria ({ifs_matrix.n_criteria})",
        )

    # Extract score matrix (IFS score function: μ − ν)
    score_matrix = ifs_matrix.score_matrix()  # shape (n_prov, n_crit)

    # Compute preference matrix: P(i, k) for all pairs (i, k)
    pref_matrix = _compute_preference_matrix(
        score_matrix, weights, p_parameter
    )  # shape (n_prov, n_prov)

    # Compute positive and negative flows
    phi_plus, phi_minus, phi_net = _compute_flows(pref_matrix)

    # Rank by net flow descending
    ranks = _score_to_rank(phi_net)

    return RankingResult(
        method=RankingMethod.IF_PROMETHEE2,
        year=ifs_matrix.year,
        provinces=ifs_matrix.alternatives,
        scores=phi_net.tolist(),
        ranks=ranks.tolist(),
    )


# =============================================================================
# Internal helpers
# =============================================================================

def _compute_preference_matrix(
    score_matrix: np.ndarray,
    weights: np.ndarray,
    p: float,
) -> np.ndarray:
    """
    Compute weighted preference matrix P(i, k) for all province pairs.

    π(i, k) = ∑_j w_j · P_j(i, k)

    where P_j(i, k) = preference_gaussian(S_ij − S_kj, p)

    Parameters
    ----------
    score_matrix : ndarray, shape (n_prov, n_crit)
        IFS score function values (μ − ν).
    weights : ndarray, shape (n_crit,)
        Criterion weights.
    p : float > 0
        Gaussian preference parameter.

    Returns
    -------
    pref_matrix : ndarray, shape (n_prov, n_prov)
        pref_matrix[i, k] = π(i, k)
    """
    n_prov = score_matrix.shape[0]
    n_crit = score_matrix.shape[1]

    # Normalise weights across non-NaN criteria
    w = np.asarray(weights, dtype=float)

    pref_matrix = np.zeros((n_prov, n_prov), dtype=float)

    # For each pair (i, k)
    for i in range(n_prov):
        for k in range(n_prov):
            if i == k:
                pref_matrix[i, k] = 0.0  # No preference vs self
                continue

            # Aggregate preferences across criteria
            pref_sum = 0.0
            w_sum = 0.0

            for j in range(n_crit):
                s_ij = score_matrix[i, j]
                s_kj = score_matrix[k, j]

                # Skip if either is NaN
                if np.isnan(s_ij) or np.isnan(s_kj):
                    continue

                w_j = w[j]
                d = s_ij - s_kj
                p_j = preference_gaussian(d, p)
                pref_sum += w_j * p_j
                w_sum += w_j

            # Renormalise by active weight sum
            if w_sum > _TOL:
                pref_matrix[i, k] = pref_sum / w_sum
            else:
                pref_matrix[i, k] = 0.0

    return pref_matrix


def preference_gaussian(d: float, p: float) -> float:
    """
    Gaussian preference function.

    P(d) = 0               if d ≤ 0
         = 1 − exp(−d²/(2p²))  if d > 0

    Parameters
    ----------
    d : float
        Difference in scores.
    p : float > 0
        Shape parameter.

    Returns
    -------
    pref : float ∈ [0, 1]
    """
    if d <= 0.0:
        return 0.0
    return 1.0 - np.exp(-d ** 2 / (2.0 * p ** 2))


def _compute_flows(pref_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute positive, negative, and net outranking flows.

    φ⁺(i) = (1/(n−1)) · ∑_k π(i, k)  — how strongly i outranks others
    φ⁻(i) = (1/(n−1)) · ∑_k π(k, i)  — how strongly others outrank i
    φ(i) = φ⁺(i) − φ⁻(i)              — net flow

    Parameters
    ----------
    pref_matrix : ndarray, shape (n_prov, n_prov)

    Returns
    -------
    phi_plus, phi_minus, phi_net : ndarray, shape (n_prov,)
    """
    n = pref_matrix.shape[0]

    if n <= 1:
        # Trivial case: single province
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    # φ⁺(i) = row sum / (n−1)
    phi_plus = np.sum(pref_matrix, axis=1) / (n - 1)

    # φ⁻(i) = column sum / (n−1)
    phi_minus = np.sum(pref_matrix, axis=0) / (n - 1)

    # φ(i) = φ⁺(i) − φ⁻(i)
    phi_net = phi_plus - phi_minus

    return phi_plus, phi_minus, phi_net


def _score_to_rank(scores: np.ndarray) -> np.ndarray:
    """
    Convert net flow to rank (1-indexed, higher net flow = better rank).

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
