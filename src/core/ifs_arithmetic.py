"""
src/core/ifs_arithmetic.py
--------------------------
Intuitionistic Fuzzy Set (IFS) arithmetic primitives.

Mathematical foundation
-----------------------
An IFS number A = (μ, ν, π) satisfies:
    μ ≥ 0,  ν ≥ 0,  π ≥ 0,  μ + ν + π = 1
where
    μ  = degree of membership
    ν  = degree of non-membership
    π  = degree of hesitancy  (always derived as 1 - μ - ν)

All operations follow Atanassov (1986, 1999) and Xu & Yager (2006).

Vectorised batch operations use NumPy arrays of shape (..., n_criteria) for
μ, ν, π separately, which is the efficient representation used by the MCDM
pipeline.  The scalar ``IFSNumber`` class is provided for conceptual clarity,
unit tests, and small-scale computations.

References
----------
Atanassov, K.T. (1986). Intuitionistic fuzzy sets. Fuzzy Sets Syst. 20, 87-96.
Xu, Z., Yager, R.R. (2006). Some geometric aggregation operators based on
    intuitionistic fuzzy sets. Int. J. Gen. Syst. 35(4), 417-433.
Boran, F.E. et al. (2009). A multi-criteria intuitionistic fuzzy group
    decision making for supplier selection with TOPSIS method.
    Expert Syst. Appl. 36(8), 11363-11368.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from src.core.exceptions import IFSArithmeticError, IFSValueError

# ---------------------------------------------------------------------------
# Numerical tolerances
# ---------------------------------------------------------------------------
_TOL: float = 1e-9          # tolerance for constraint checking
_CLIP_EPS: float = 0.0      # minimum clipped value (strict non-negativity)


# =============================================================================
# IFSNumber — scalar IFS triple
# =============================================================================

class IFSNumber:
    """
    Immutable scalar Intuitionistic Fuzzy Number A = (μ, ν, π).

    π is always derived as 1 − μ − ν to guarantee the partition-of-unity
    constraint exactly.  An explicitly supplied π is validated within ``_TOL``.

    Parameters
    ----------
    mu : float  — membership degree ∈ [0, 1]
    nu : float  — non-membership degree ∈ [0, 1]
    pi : float, optional — hesitancy; computed as 1−μ−ν if omitted.

    Raises
    ------
    IFSValueError
        If μ < 0, ν < 0, or μ + ν > 1.
    """

    __slots__ = ("mu", "nu", "pi")

    def __init__(self, mu: float, nu: float, pi: Optional[float] = None) -> None:
        mu_f = float(mu)
        nu_f = float(nu)
        pi_computed = 1.0 - mu_f - nu_f

        if pi is not None:
            pi_f = float(pi)
            if abs(pi_f - pi_computed) > _TOL:
                raise IFSValueError(
                    f"Supplied π={pi_f:.6f} ≠ 1−μ−ν={pi_computed:.6f}",
                    context={"mu": mu_f, "nu": nu_f, "pi_supplied": pi_f,
                             "pi_computed": pi_computed},
                )
        else:
            pi_f = pi_computed

        if mu_f < -_TOL or nu_f < -_TOL or pi_f < -_TOL:
            raise IFSValueError(
                f"IFS components must be ≥ 0: μ={mu_f:.4f}, ν={nu_f:.4f}, π={pi_f:.4f}",
                context={"mu": mu_f, "nu": nu_f, "pi": pi_f},
            )
        if mu_f + nu_f > 1.0 + _TOL:
            raise IFSValueError(
                f"μ + ν = {mu_f + nu_f:.6f} > 1",
                context={"mu": mu_f, "nu": nu_f},
            )

        # Clip tiny floating-point negatives, recompute pi from clipped mu/nu
        mu_f = max(0.0, mu_f)
        nu_f = max(0.0, nu_f)
        pi_f = max(0.0, 1.0 - mu_f - nu_f)

        object.__setattr__(self, "mu", mu_f)
        object.__setattr__(self, "nu", nu_f)
        object.__setattr__(self, "pi", pi_f)

    # Make immutable
    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("IFSNumber is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("IFSNumber is immutable")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IFSNumber):
            return NotImplemented
        return (
            math.isclose(self.mu, other.mu, abs_tol=_TOL)
            and math.isclose(self.nu, other.nu, abs_tol=_TOL)
        )

    def __hash__(self) -> int:
        return hash((round(self.mu, 9), round(self.nu, 9)))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"IFS(μ={self.mu:.4f}, ν={self.nu:.4f}, π={self.pi:.4f})"

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.mu, self.nu, self.pi)

    def score(self) -> float:
        """Score function S(A) = μ − ν ∈ [−1, 1]."""
        return score_function(self)

    def accuracy(self) -> float:
        """Accuracy function H(A) = μ + ν ∈ [0, 1]."""
        return accuracy_function(self)


def _ifs(mu: float, nu: float) -> IFSNumber:
    """Internal factory: create IFSNumber from (μ, ν); π is derived."""
    mu_c = max(0.0, min(1.0, float(mu)))
    nu_c = max(0.0, min(1.0 - mu_c, float(nu)))
    return IFSNumber(mu_c, nu_c)


# =============================================================================
# Scalar IFS operations
# =============================================================================

def ifs_add(a: IFSNumber, b: IFSNumber) -> IFSNumber:
    """
    IFS algebraic sum (union): A ⊕ B.

    A ⊕ B = (μ_A + μ_B − μ_A·μ_B,  ν_A·ν_B,  derived π)

    References: Atanassov (1986), Definition 3.
    """
    mu = a.mu + b.mu - a.mu * b.mu
    nu = a.nu * b.nu
    return _ifs(mu, nu)


def ifs_multiply(a: IFSNumber, b: IFSNumber) -> IFSNumber:
    """
    IFS algebraic product (intersection): A ⊗ B.

    A ⊗ B = (μ_A·μ_B,  ν_A + ν_B − ν_A·ν_B,  derived π)

    References: Atanassov (1986), Definition 3.
    """
    mu = a.mu * b.mu
    nu = a.nu + b.nu - a.nu * b.nu
    return _ifs(mu, nu)


def ifs_scalar_multiply(a: IFSNumber, lam: float) -> IFSNumber:
    """
    Scalar (lambda) multiplication: λA, for λ > 0.

    λA = (1 − (1−μ_A)^λ,  ν_A^λ,  derived π)

    References: Xu & Yager (2006), Definition 1.

    Raises
    ------
    IFSArithmeticError
        If λ ≤ 0.
    """
    if lam <= 0:
        raise IFSArithmeticError(
            "Scalar multiplier λ must be > 0",
            context={"lambda": lam},
        )
    mu = 1.0 - (1.0 - a.mu) ** lam
    nu = a.nu ** lam
    return _ifs(mu, nu)


def ifs_power(a: IFSNumber, lam: float) -> IFSNumber:
    """
    IFS power: A^λ, for λ > 0.

    A^λ = (μ_A^λ,  1 − (1−ν_A)^λ,  derived π)

    References: Xu & Yager (2006), Definition 1.

    Raises
    ------
    IFSArithmeticError
        If λ ≤ 0.
    """
    if lam <= 0:
        raise IFSArithmeticError(
            "Power exponent λ must be > 0",
            context={"lambda": lam},
        )
    mu = a.mu ** lam
    nu = 1.0 - (1.0 - a.nu) ** lam
    return _ifs(mu, nu)


# =============================================================================
# Aggregation operators (scalar, closed-form)
# =============================================================================

def ifs_wam(
    ifs_list: Sequence[IFSNumber],
    weights: Sequence[float],
) -> IFSNumber:
    """
    Intuitionistic Fuzzy Weighted Arithmetic Mean (IFWAM).

    Closed-form for n elements with weights w_j summing to 1:

        μ_agg = 1 − ∏(1 − μ_j)^w_j
        ν_agg = ∏ν_j^w_j
        π_agg = ∏(1 − μ_j)^w_j − ∏ν_j^w_j

    Equivalent to:  ⊕_{j=1}^{n} (w_j · A_j)

    Parameters
    ----------
    ifs_list : sequence of IFSNumber
        The IFS values to aggregate.
    weights : sequence of float
        Non-negative weights.  Need not sum to 1 — they are normalised
        internally.

    Raises
    ------
    IFSArithmeticError
        If the weight vector sums to zero or lengths mismatch.
    """
    n = len(ifs_list)
    if len(weights) != n:
        raise IFSArithmeticError(
            f"Length mismatch: {n} IFS values but {len(weights)} weights",
        )
    w = np.asarray(weights, dtype=float)
    if w.sum() < _TOL:
        raise IFSArithmeticError("Weights sum to zero — cannot normalise")
    w = w / w.sum()

    mu_arr = np.array([a.mu for a in ifs_list], dtype=float)
    nu_arr = np.array([a.nu for a in ifs_list], dtype=float)

    # ∏(1 − μ_j)^w_j  and  ∏ν_j^w_j
    prod_1m_mu = np.prod((1.0 - mu_arr) ** w)
    prod_nu = np.prod(nu_arr ** w)

    mu_agg = 1.0 - prod_1m_mu
    nu_agg = prod_nu
    return _ifs(mu_agg, nu_agg)


def ifs_wgm(
    ifs_list: Sequence[IFSNumber],
    weights: Sequence[float],
) -> IFSNumber:
    """
    Intuitionistic Fuzzy Weighted Geometric Mean (IFWGM).

    Closed-form for n elements with weights w_j summing to 1:

        μ_agg = ∏μ_j^w_j
        ν_agg = 1 − ∏(1 − ν_j)^w_j
        π_agg = ∏(1 − ν_j)^w_j − ∏μ_j^w_j

    Equivalent to:  ⊗_{j=1}^{n} A_j^w_j

    Parameters
    ----------
    ifs_list : sequence of IFSNumber
    weights : sequence of float
        Non-negative weights, normalised internally.

    Raises
    ------
    IFSArithmeticError
        If weight sum is zero or lengths mismatch.
    """
    n = len(ifs_list)
    if len(weights) != n:
        raise IFSArithmeticError(
            f"Length mismatch: {n} IFS values but {len(weights)} weights",
        )
    w = np.asarray(weights, dtype=float)
    if w.sum() < _TOL:
        raise IFSArithmeticError("Weights sum to zero — cannot normalise")
    w = w / w.sum()

    mu_arr = np.array([a.mu for a in ifs_list], dtype=float)
    nu_arr = np.array([a.nu for a in ifs_list], dtype=float)

    prod_mu = np.prod(mu_arr ** w)
    prod_1m_nu = np.prod((1.0 - nu_arr) ** w)

    mu_agg = prod_mu
    nu_agg = 1.0 - prod_1m_nu
    return _ifs(mu_agg, nu_agg)


# =============================================================================
# Vectorised aggregation (NumPy batch) — used by MCDM pipeline
# =============================================================================

def vec_wam(
    mu: np.ndarray,
    nu: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised IFWAM over the last axis (criteria axis).

    Parameters
    ----------
    mu, nu : ndarray, shape (..., n_criteria)
        Membership and non-membership arrays.  NaN values are excluded
        from the weighted product via weight re-normalisation per row.
    weights : ndarray, shape (n_criteria,)
        Non-negative raw weights (normalised internally per row, accounting
        for NaN positions).

    Returns
    -------
    mu_agg, nu_agg, pi_agg : ndarray, shape (...)
    """
    w = np.asarray(weights, dtype=float)
    # Broadcast weights to match leading dimensions; mask NaN positions
    valid = ~(np.isnan(mu) | np.isnan(nu))               # (..., n_criteria)
    w_broad = np.where(valid, w, 0.0)                     # zero-out NaN cols
    w_sum = w_broad.sum(axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        w_norm = np.where(w_sum > _TOL, w_broad / w_sum, 0.0)

    # Replace NaN with neutral values for product: (1-mu)=1 when excluded, nu=1 when excluded
    mu_safe = np.where(valid, mu, 0.0)
    nu_safe = np.where(valid, nu, 1.0)

    prod_1m_mu = np.prod(np.where(w_norm > 0, (1.0 - mu_safe) ** w_norm, 1.0), axis=-1)
    prod_nu    = np.prod(np.where(w_norm > 0, nu_safe ** w_norm, 1.0), axis=-1)

    mu_agg = np.clip(1.0 - prod_1m_mu, 0.0, 1.0)
    nu_agg = np.clip(prod_nu, 0.0, 1.0)
    pi_agg = np.clip(1.0 - mu_agg - nu_agg, 0.0, 1.0)
    return mu_agg, nu_agg, pi_agg


def vec_wgm(
    mu: np.ndarray,
    nu: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised IFWGM over the last axis (criteria axis).

    Parameters
    ----------
    mu, nu : ndarray, shape (..., n_criteria)
    weights : ndarray, shape (n_criteria,)

    Returns
    -------
    mu_agg, nu_agg, pi_agg : ndarray, shape (...)
    """
    w = np.asarray(weights, dtype=float)
    valid = ~(np.isnan(mu) | np.isnan(nu))
    w_broad = np.where(valid, w, 0.0)
    w_sum = w_broad.sum(axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        w_norm = np.where(w_sum > _TOL, w_broad / w_sum, 0.0)

    mu_safe = np.where(valid, mu, 1.0)   # neutral for product: mu=1
    nu_safe = np.where(valid, nu, 0.0)   # neutral for product: nu=0

    prod_mu    = np.prod(np.where(w_norm > 0, mu_safe ** w_norm, 1.0), axis=-1)
    prod_1m_nu = np.prod(np.where(w_norm > 0, (1.0 - nu_safe) ** w_norm, 1.0), axis=-1)

    mu_agg = np.clip(prod_mu, 0.0, 1.0)
    nu_agg = np.clip(1.0 - prod_1m_nu, 0.0, 1.0)
    pi_agg = np.clip(1.0 - mu_agg - nu_agg, 0.0, 1.0)
    return mu_agg, nu_agg, pi_agg


# =============================================================================
# Scalar functions
# =============================================================================

def score_function(a: IFSNumber) -> float:
    """
    Score function: S(A) = μ − ν ∈ [−1, 1].

    Higher score → preferred alternative.

    References: Chen & Tan (1994).
    """
    return a.mu - a.nu


def accuracy_function(a: IFSNumber) -> float:
    """
    Accuracy function: H(A) = μ + ν ∈ [0, 1].

    Used as tie-breaker in :func:`ifs_compare`.

    References: Hong & Choi (2000).
    """
    return a.mu + a.nu


def ifs_compare(a: IFSNumber, b: IFSNumber) -> int:
    """
    Compare two IFS numbers using score then accuracy.

    Returns
    -------
    int
        +1 if A > B,  −1 if A < B,  0 if A == B.

    Decision rules (Chen & Tan 1994; Hong & Choi 2000):
    1. If S(A) > S(B) → A > B.
    2. If S(A) == S(B) and H(A) > H(B) → A > B.
    3. If S(A) == S(B) and H(A) == H(B) → A == B.
    """
    sa, sb = score_function(a), score_function(b)
    if sa - sb > _TOL:
        return 1
    if sb - sa > _TOL:
        return -1
    # Tie-break by accuracy
    ha, hb = accuracy_function(a), accuracy_function(b)
    if ha - hb > _TOL:
        return 1
    if hb - ha > _TOL:
        return -1
    return 0


# =============================================================================
# Distance measures
# =============================================================================

def hamming_distance(a: IFSNumber, b: IFSNumber) -> float:
    """
    Normalised Hamming distance between two IFS numbers ∈ [0, 1].

    d_H(A, B) = ½(|μ_A − μ_B| + |ν_A − ν_B| + |π_A − π_B|)

    References: Szmidt & Kacprzyk (2000).
    """
    return 0.5 * (abs(a.mu - b.mu) + abs(a.nu - b.nu) + abs(a.pi - b.pi))


def normalized_euclidean_distance(a: IFSNumber, b: IFSNumber) -> float:
    """
    Normalised Euclidean distance between two IFS numbers ∈ [0, 1].

    d_NE(A, B) = √(½((μ_A−μ_B)² + (ν_A−ν_B)² + (π_A−π_B)²))

    References: Szmidt & Kacprzyk (2000).
    """
    return math.sqrt(
        0.5 * (
            (a.mu - b.mu) ** 2
            + (a.nu - b.nu) ** 2
            + (a.pi - b.pi) ** 2
        )
    )


def vec_normalized_euclidean_distance(
    mu1: np.ndarray,
    nu1: np.ndarray,
    pi1: np.ndarray,
    mu2: np.ndarray,
    nu2: np.ndarray,
    pi2: np.ndarray,
) -> np.ndarray:
    """
    Vectorised normalised Euclidean distance between two IFS arrays.

    All arrays must be broadcastable to the same shape.

    Returns
    -------
    ndarray
        Distance array of the broadcast shape.
    """
    return np.sqrt(
        0.5 * (
            (mu1 - mu2) ** 2
            + (nu1 - nu2) ** 2
            + (pi1 - pi2) ** 2
        )
    )


# =============================================================================
# IFS conversion from raw scores
# =============================================================================

def score_to_ifs(
    x: float,
    x_max: float,
    pi_fixed: float = 0.05,
    method: str = "fixed_pi",
) -> IFSNumber:
    """
    Convert a raw governance score to an IFS number.

    Fixed-π method (default)
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Given x ∈ [0, x_max] and a constant hesitancy π₀:

        μ = (x / x_max) × (1 − π₀)
        ν = (1 − x / x_max) × (1 − π₀)
        π = π₀

    Verification: μ + ν + π = (1−π₀) + π₀ = 1  ✓

    Properties:
    * When x = x_max → μ = 1−π₀,  ν = 0,    π = π₀
    * When x = 0     → μ = 0,      ν = 1−π₀, π = π₀
    * All sub-criteria are benefit-type (higher score → higher μ)

    Parameters
    ----------
    x : float
        Raw score.  NaN is propagated (returns NaN triple via convention).
    x_max : float
        Theoretical maximum score (3.33 for PAPI).
    pi_fixed : float
        Fixed hesitancy π₀ ∈ (0, 1).  Default 0.05.
    method : str
        Conversion method.  Only ``"fixed_pi"`` is currently supported.

    Returns
    -------
    IFSNumber

    Raises
    ------
    IFSArithmeticError
        If x_max ≤ 0 or pi_fixed not in (0, 1).
    """
    if x_max <= 0:
        raise IFSArithmeticError("x_max must be > 0", context={"x_max": x_max})
    if not (0.0 <= pi_fixed < 1.0):
        raise IFSArithmeticError(
            "pi_fixed must be in [0, 1)",
            context={"pi_fixed": pi_fixed},
        )
    if method != "fixed_pi":
        raise IFSArithmeticError(
            f"Unknown conversion method '{method}'",
            context={"method": method},
        )

    if math.isnan(x):
        # Propagate NaN — caller must handle
        return IFSNumber(float("nan"), float("nan"), float("nan"))  # type: ignore

    x_norm = float(np.clip(x / x_max, 0.0, 1.0))
    scale = 1.0 - pi_fixed
    mu = x_norm * scale
    nu = (1.0 - x_norm) * scale
    return _ifs(mu, nu)


def vec_score_to_ifs(
    x_arr: np.ndarray,
    x_max: float,
    pi_fixed: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised conversion of raw scores to IFS triples.

    Parameters
    ----------
    x_arr : ndarray, any shape
        Raw scores.  NaN values produce NaN in output.
    x_max : float
        Theoretical maximum.
    pi_fixed : float
        Fixed hesitancy.

    Returns
    -------
    mu, nu, pi : ndarray
        Same shape as ``x_arr``.
    """
    if x_max <= 0:
        raise IFSArithmeticError("x_max must be > 0", context={"x_max": x_max})

    x_norm = np.clip(x_arr / x_max, 0.0, 1.0)
    scale = 1.0 - pi_fixed
    mu = x_norm * scale
    nu = (1.0 - x_norm) * scale
    pi = np.full_like(mu, pi_fixed)
    # Propagate NaN from input
    nan_mask = np.isnan(x_arr)
    mu[nan_mask] = np.nan
    nu[nan_mask] = np.nan
    pi[nan_mask] = np.nan
    return mu, nu, pi


# =============================================================================
# IFSMatrix — batch IFS representation for a province × sub-criteria panel
# =============================================================================

@dataclass
class IFSMatrix:
    """
    IFS representation of a single year's province × sub-criteria panel.

    All three arrays have shape ``(n_alternatives, n_criteria)``.
    NaN entries represent missing sub-criteria for a given province-year.

    Attributes
    ----------
    mu : ndarray, shape (n_alts, n_crit)
    nu : ndarray, shape (n_alts, n_crit)
    pi : ndarray, shape (n_alts, n_crit)
    alternatives : list[str]
        Province codes (rows).
    criteria : list[str]
        Sub-criteria codes (columns).
    year : int
    """

    mu: np.ndarray
    nu: np.ndarray
    pi: np.ndarray
    alternatives: List[str]
    criteria: List[str]
    year: int

    def __post_init__(self) -> None:
        shapes = {self.mu.shape, self.nu.shape, self.pi.shape}
        if len(shapes) != 1:
            raise IFSArithmeticError("mu, nu, pi must have identical shapes")
        n_alts, n_crit = self.mu.shape
        if n_alts != len(self.alternatives):
            raise IFSArithmeticError(
                f"mu rows ({n_alts}) ≠ len(alternatives) ({len(self.alternatives)})"
            )
        if n_crit != len(self.criteria):
            raise IFSArithmeticError(
                f"mu cols ({n_crit}) ≠ len(criteria) ({len(self.criteria)})"
            )

    @property
    def n_alternatives(self) -> int:
        return len(self.alternatives)

    @property
    def n_criteria(self) -> int:
        return len(self.criteria)

    def score_matrix(self) -> np.ndarray:
        """Return S = μ − ν, shape (n_alts, n_crit)."""
        return self.mu - self.nu

    def get_criterion_index(self, criterion: str) -> int:
        try:
            return self.criteria.index(criterion)
        except ValueError as exc:
            raise KeyError(f"Criterion '{criterion}' not in IFSMatrix") from exc


def ifs_matrix_from_dataframe(
    df: "pd.DataFrame",
    x_max: float,
    pi_fixed: float = 0.05,
    year: int = 0,
) -> IFSMatrix:
    """
    Build an :class:`IFSMatrix` from a province-indexed raw-score DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Province-indexed DataFrame (rows = provinces, cols = sub-criteria).
        NaN values in the DataFrame are preserved as NaN in the IFS arrays.
    x_max : float
        Theoretical maximum score (3.33 for PAPI).
    pi_fixed : float
        Fixed hesitancy π₀.
    year : int
        Calendar year (for metadata).

    Returns
    -------
    IFSMatrix
    """
    import pandas as pd  # local import to avoid circular dependency at module level

    values = df.to_numpy(dtype=float)   # shape (n_alts, n_crit)
    mu, nu, pi = vec_score_to_ifs(values, x_max=x_max, pi_fixed=pi_fixed)

    return IFSMatrix(
        mu=mu,
        nu=nu,
        pi=pi,
        alternatives=list(df.index),
        criteria=list(df.columns),
        year=year,
    )
