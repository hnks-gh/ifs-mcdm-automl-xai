"""
src/core/exceptions.py
-----------------------
Custom exception hierarchy for the IFS-MCDM-AutoML-XAI framework.

All framework exceptions derive from ``FrameworkError``, allowing callers to
catch the entire hierarchy with a single ``except FrameworkError`` clause while
still being able to target specific sub-types when needed.
"""

from __future__ import annotations


# =============================================================================
# Base
# =============================================================================

class FrameworkError(Exception):
    """
    Base class for all IFS-MCDM-AutoML-XAI framework errors.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    context : dict | None
        Optional structured context (e.g. year, province, column name) that
        aids debugging without relying on string parsing.
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict = context or {}

    def __str__(self) -> str:  # noqa: D105
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message


# =============================================================================
# Configuration
# =============================================================================

class ConfigurationError(FrameworkError):
    """
    Raised when a required configuration key is missing, has an invalid type,
    or contains a logically inconsistent value.

    Example
    -------
    >>> raise ConfigurationError(
    ...     "lambda_param must be in [0, 1]",
    ...     context={"lambda_param": 1.5, "method": "if_waspas"},
    ... )
    """


# =============================================================================
# Data Layer
# =============================================================================

class DataLoadError(FrameworkError):
    """
    Raised when a CSV file cannot be located, read, or parsed.

    Typical causes: missing file, malformed CSV, wrong encoding.

    Context keys typically include: ``year``, ``path``.
    """


class DataIntegrityError(FrameworkError):
    """
    Raised when loaded data fails schema or integrity validation.

    Examples: unexpected number of provinces, unrecognised column names,
    duplicate province codes, values outside the expected range [0, 3.33].

    Context keys typically include: ``year``, ``expected``, ``found``.
    """


class RegimeDetectionError(FrameworkError):
    """
    Raised when the automatic regime detection algorithm cannot determine a
    consistent regime for a given year.

    Context keys typically include: ``year``, ``detected_regime``.
    """


# =============================================================================
# IFS Layer
# =============================================================================

class IFSValueError(FrameworkError):
    """
    Raised when an IFS triple (μ, ν, π) violates the fundamental constraints:
    μ ≥ 0, ν ≥ 0, π ≥ 0, and μ + ν + π = 1.

    Context keys typically include: ``mu``, ``nu``, ``pi``, ``source``.
    """


class IFSArithmeticError(FrameworkError):
    """
    Raised when an IFS arithmetic operation produces an invalid result
    (e.g. out-of-range aggregation due to floating-point accumulation).

    Context keys typically include: ``operation``, ``operands``.
    """


# =============================================================================
# MCDM Layer
# =============================================================================

class MCDMError(FrameworkError):
    """
    Generic base for all MCDM computation errors.

    Context keys typically include: ``method``, ``year``, ``stage``.
    """


class WeightingError(MCDMError):
    """
    Raised when the IF-CRITIC weighting computation fails.

    Typical causes: all-zero variance columns after regime filtering,
    singular correlation matrix.

    Context keys typically include: ``method``, ``year``, ``stage``,
    ``criterion``.
    """


class RankingError(MCDMError):
    """
    Raised when a ranking method (IF-WASPAS, IF-TOPSIS, IF-PROMETHEE II)
    fails or produces invalid output.

    Context keys typically include: ``method``, ``year``, ``n_provinces``.
    """


class AnalysisError(MCDMError):
    """
    Raised when stability or sensitivity analysis computations fail.

    Context keys typically include: ``analysis_type``, ``year``, ``window``.
    """


# =============================================================================
# ML Layer
# =============================================================================

class ImputationError(FrameworkError):
    """
    Raised when MICE imputation fails or produces unacceptable output
    (e.g. NaN values remain after imputation).

    Context keys typically include: ``n_missing_before``, ``n_missing_after``.
    """


class ForecastingError(FrameworkError):
    """
    Raised when AutoGluon TimeSeriesPredictor training or prediction fails.

    Context keys typically include: ``target_col``, ``year``, ``predictor``.
    """


class ExplainabilityError(FrameworkError):
    """
    Raised when SHAP explainability computation fails.

    Context keys typically include: ``target_col``, ``explainer_type``.
    """


# =============================================================================
# I/O
# =============================================================================

class OutputError(FrameworkError):
    """
    Raised when saving or loading output artefacts fails.

    Context keys typically include: ``path``, ``operation``.
    """
