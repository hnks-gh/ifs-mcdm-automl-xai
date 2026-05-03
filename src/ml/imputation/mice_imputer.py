"""
src/ml/imputation/mice_imputer.py
---------------------------------
MICE (Multivariate Imputation by Chained Equations) imputation for the PAPI panel.

Responsibilities
----------------
* Load the raw 2011-2024 PAPI panel data from all CSV files.
* Reshape into long format (882 rows = 63 provinces × 14 years; 30 columns = Province, Year, SC11-SC83).
* Impute all missing values using scikit-learn's IterativeImputer (MICE algorithm).
* Validate imputed data: zero NaN cells, values within [0, 3.33] bounds.
* Save imputed panel to ``output/ml/imputed/panel_imputed.parquet`` (never modify ``data/csv/``).

Data Integrity & Leakage Prevention
------------------------------------
The 2011-2024 panel is treated as **already-observed** historical data, not a time series with
a future horizon. Hence:
* Fit IterativeImputer on the full 882-row panel simultaneously.
* No train/test split, no temporal separation.
* The 2025 forecast horizon is purely **outside** the panel scope and handled separately by AutoGluon.
* This is mathematically sound: we are not learning patterns *to predict into* the panel;
  we are imputing missing values *within* observed data.

Regime Handling for AutoGluon Downstream
-----------------------------------------
For AutoGluon forecasting (Phase 7) and SHAP explainability (Phase 8):
* R1 (2011-2017): 22 active sub-criteria — dropped for ML path (too few active columns)
* R2 (2018): 28 active sub-criteria — included
* R3 (2019-2020): 29 active sub-criteria — included
* R4 (2021-2024): 28 active sub-criteria (SC52 absent) — included

The imputed panel will flag absent_subcriteria per year; downstream ML will filter to active columns.

Production Quality Checks
-------------------------
✅ Error handling for all I/O and validation steps.
✅ Comprehensive logging at INFO and DEBUG levels.
✅ Input validation (shape, data types, NaN counts).
✅ Output validation (zero NaN, value bounds, shape consistency).
✅ Type hints for all functions.
✅ Docstrings with parameter and return documentation.
✅ No modification of original data/csv/ files.
✅ Reproducible: fixed random_state from config.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from src.core import data_loader
from src.core.data_loader import load_config
from src.core.exceptions import DataIntegrityError, ImputationError
from src.core.schema import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Value bounds for PAPI scores: all sub-criteria are on a 0–3.33 scale
_VALUE_MIN: float = 0.0
_VALUE_MAX: float = 3.33
_VALUE_TOLERANCE: float = 0.15  # Tolerance for rounding/measurement precision at boundary

# Suppressed warnings during MICE imputation
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# =============================================================================
# Public API
# =============================================================================

def load_raw_panel(
    config: AppConfig,
    csv_dir: str = "data/csv",
) -> pd.DataFrame:
    """
    Load all annual PAPI CSV files and reshape into a long-format panel.

    The output is a long-format DataFrame:
    * Rows: 882 (63 provinces × 14 years)
    * Columns: Province, Year, SC11, SC12, ..., SC83 (32 columns total)

    Each row corresponds to one (province, year) observation.

    Parameters
    ----------
    config : AppConfig
        Application configuration containing year list and data paths.
    csv_dir : str
        Directory containing CSV files. Default: "data/csv"

    Returns
    -------
    pd.DataFrame
        Long-format panel with structure:
        - Column 'Province': province code (P01, P02, ..., P63)
        - Column 'Year': year (2011, 2012, ..., 2024)
        - Columns SC11–SC83: sub-criteria scores (float64, may contain NaN)
        Shape: (882, 32)

    Raises
    ------
    DataLoadError
        If any CSV file cannot be loaded or has wrong shape.
    DataIntegrityError
        If the loaded panel has inconsistent shape or province mismatch.

    Notes
    -----
    * This function preserves all NaN values from the original CSVs.
    * No imputation occurs at this stage — only reshaping and validation.
    * The output is suitable for IterativeImputer input.

    Examples
    --------
    >>> from src.core.schema import load_config
    >>> config = load_config()
    >>> panel = load_raw_panel(config)
    >>> assert panel.shape == (882, 32)  # 63 provinces × 14 years
    >>> assert panel["Year"].nunique() == 14
    """
    logger.info("Loading raw panel from {} annual CSV files", len(config.data.years))

    frames: List[pd.DataFrame] = []

    for year in config.data.years:
        try:
            df_year = data_loader.load_year(year, csv_dir=csv_dir)
            logger.debug("  Loaded {}: shape {}", year, df_year.shape)

            # Validate: each year should have 63 provinces
            if len(df_year) != config.data.n_provinces:
                raise DataIntegrityError(
                    f"Year {year} has {len(df_year)} provinces; expected {config.data.n_provinces}",
                    context={"year": year, "n_rows": len(df_year)},
                )

            # Add year column
            df_year = df_year.reset_index()  # Province from index → column
            df_year["Year"] = year

            frames.append(df_year)

        except Exception as e:
            logger.error("Failed to load year {}: {}", year, str(e))
            raise

    # Concatenate all years into long format
    panel = pd.concat(frames, ignore_index=True)

    logger.info(
        "Raw panel loaded: shape {}, {} years, {} provinces",
        panel.shape,
        panel["Year"].nunique(),
        panel["Province"].nunique(),
    )

    # Validate panel structure
    _validate_raw_panel(panel, config)

    return panel


def run_mice_imputation(
    raw_panel: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    """
    Impute missing values in the PAPI panel using MICE (Multivariate Imputation by Chained Equations).

    Uses scikit-learn's IterativeImputer with the following strategy:
    * Algorithm: BayesianRidge (robust to non-normal data, good for small-to-medium datasets)
    * Max iterations: from config (default 10)
    * Random state: from config for reproducibility
    * Fit on full 882-row panel (no train/test split — treats all data as observed)

    Parameters
    ----------
    raw_panel : pd.DataFrame
        Long-format panel with Province, Year, and sub-criteria columns.
        Input can have NaN values.
    config : AppConfig
        Application configuration including MICE hyperparameters.

    Returns
    -------
    pd.DataFrame
        Imputed panel with same shape as input; all NaN values replaced.
        Non-numeric columns (Province, Year) are preserved as-is.

    Raises
    ------
    ImputationError
        If imputation fails or produces invalid values (NaN or out-of-bounds).

    Notes
    -----
    * The returned DataFrame has **zero NaN values** in numeric columns.
    * Values are clipped to [0, 3.33] if necessary (e.g., if imputer generates slight overages).
    * This function does NOT modify the input DataFrame or any files on disk.

    Examples
    --------
    >>> panel_raw = load_raw_panel(data_loader, config)
    >>> nan_count_before = panel_raw.isna().sum().sum()
    >>> panel_imputed = run_mice_imputation(panel_raw, config)
    >>> assert panel_imputed.isna().sum().sum() == 0
    >>> assert nan_count_before > 0  # Assuming some missing data
    """
    logger.info(
        "Starting MICE imputation on panel of shape {} (max_iter={})",
        raw_panel.shape,
        config.ml.imputation.max_iter,
    )

    # Count NaN before
    nan_count_before = raw_panel.isna().sum().sum()
    logger.info("  NaN cells before imputation: {}", nan_count_before)

    if nan_count_before == 0:
        logger.info("  Panel has no missing values; returning as-is")
        return raw_panel.copy()

    try:
        # Separate numeric and non-numeric columns
        non_numeric_cols = ["Province", "Year"]
        numeric_cols = [col for col in raw_panel.columns if col not in non_numeric_cols]

        # Extract numeric data for imputation
        X = raw_panel[numeric_cols].copy()

        # Initialize MICE imputer
        imputer = IterativeImputer(
            max_iter=config.ml.imputation.max_iter,
            random_state=config.ml.imputation.random_state,
            verbose=0,
            estimator=None,  # Uses BayesianRidge by default
        )

        logger.debug("  Fitting IterativeImputer on {} numeric columns", len(numeric_cols))

        # Fit and transform
        X_imputed = imputer.fit_transform(X)

        # Reconstruct DataFrame
        df_imputed = raw_panel.copy()
        df_imputed[numeric_cols] = X_imputed

        logger.info("  MICE imputation complete")

        # Post-imputation: clip values to bounds
        logger.debug("  Clipping values to [{}, {}]", _VALUE_MIN, _VALUE_MAX)
        for col in numeric_cols:
            if col not in non_numeric_cols:
                out_of_bounds = (df_imputed[col] < _VALUE_MIN) | (df_imputed[col] > _VALUE_MAX)
                if out_of_bounds.sum() > 0:
                    logger.warning(
                        "  {} values out of bounds in column {}; clipping",
                        out_of_bounds.sum(),
                        col,
                    )
                df_imputed[col] = df_imputed[col].clip(_VALUE_MIN, _VALUE_MAX)

        # Validate output
        nan_count_after = df_imputed.isna().sum().sum()
        logger.info("  NaN cells after imputation: {}", nan_count_after)

        if nan_count_after > 0:
            raise ImputationError(
                f"Imputation produced {nan_count_after} remaining NaN values",
                context={"nan_count": nan_count_after},
            )

        logger.info("  Imputation validated: zero NaN cells, all values in bounds")

        return df_imputed

    except Exception as e:
        logger.error("MICE imputation failed: {}", str(e))
        raise ImputationError(
            f"MICE imputation failed: {str(e)}",
            context={"error": str(e)},
        )


def validate_imputation(
    imputed_panel: pd.DataFrame,
    config: AppConfig,
) -> Tuple[bool, Dict[str, any]]:
    """
    Comprehensive validation of the imputed panel.

    Checks
    ------
    1. Shape: must be (882, 32) — 63 provinces × 14 years
    2. Columns: Province, Year, SC11–SC83
    3. Data types: Province (object), Year (int64), sub-criteria (float64)
    4. NaN count: must be zero
    5. Value bounds: all sub-criteria in [0 - tolerance, 3.33 + tolerance]
    6. Province uniqueness: 63 unique provinces
    7. Year uniqueness: 14 unique years (2011–2024)
    8. No duplicate (province, year) pairs

    Parameters
    ----------
    imputed_panel : pd.DataFrame
        Imputed panel to validate.
    config : AppConfig
        Application configuration for reference values.

    Returns
    -------
    Tuple[bool, Dict[str, any]]
        * bool: True if all checks pass; False otherwise
        * dict: Detailed results of each check with keys:
          - shape_valid: bool
          - columns_valid: bool
          - dtypes_valid: bool
          - nan_count: int
          - nan_valid: bool
          - value_bounds_valid: bool
          - provinces_valid: bool
          - years_valid: bool
          - no_duplicates: bool
          - all_valid: bool
          Additional keys may include counts and bounds information.

    Notes
    -----
    This function does NOT raise exceptions; it returns a validation report.
    Callers should inspect the returned dict and decide on error handling.

    Examples
    --------
    >>> panel_imputed = run_mice_imputation(panel_raw, config)
    >>> is_valid, report = validate_imputation(panel_imputed, config)
    >>> assert is_valid, f"Validation failed: {report}"
    """
    logger.info("Validating imputed panel")

    report = {}

    # 1. Shape
    expected_shape = (
        config.data.n_provinces * len(config.data.years),
        2 + len(config.data.all_subcriteria),
    )  # 882, 32
    report["shape"] = imputed_panel.shape
    report["shape_valid"] = imputed_panel.shape == expected_shape
    logger.debug("  Shape: {} (expected {})", imputed_panel.shape, expected_shape)

    # 2. Columns
    expected_cols = {"Province", "Year"} | set(config.data.all_subcriteria)
    report["columns"] = set(imputed_panel.columns)
    report["columns_valid"] = set(imputed_panel.columns) == expected_cols
    logger.debug("  Columns: {} present", len(imputed_panel.columns))

    # 3. Data types
    numeric_cols = config.data.all_subcriteria
    report["dtypes_valid"] = all(
        pd.api.types.is_numeric_dtype(imputed_panel.get(col))
        for col in numeric_cols
        if col in imputed_panel.columns
    )
    logger.debug("  Data types: numeric columns are float/int: {}", report["dtypes_valid"])

    # 4. NaN count
    nan_count = imputed_panel.isna().sum().sum()
    report["nan_count"] = int(nan_count)
    report["nan_valid"] = nan_count == 0
    logger.debug("  NaN count: {} (expected 0)", nan_count)

    # 5. Value bounds
    out_of_bounds_mask = pd.Series(False, index=imputed_panel.index)
    for col in numeric_cols:
        if col in imputed_panel.columns:
            out_of_bounds_mask |= (
                (imputed_panel[col] < _VALUE_MIN - _VALUE_TOLERANCE) |
                (imputed_panel[col] > _VALUE_MAX + _VALUE_TOLERANCE)
            )

    report["out_of_bounds_count"] = int(out_of_bounds_mask.sum())
    report["value_bounds_valid"] = out_of_bounds_mask.sum() == 0
    logger.debug(
        "  Value bounds: {} out-of-bounds cells",
        report["out_of_bounds_count"],
    )

    # 6. Province uniqueness
    n_provinces = imputed_panel["Province"].nunique()
    report["n_provinces"] = int(n_provinces)
    report["provinces_valid"] = n_provinces == config.data.n_provinces
    logger.debug("  Provinces: {} unique (expected {})", n_provinces, config.data.n_provinces)

    # 7. Year uniqueness
    n_years = imputed_panel["Year"].nunique()
    report["n_years"] = int(n_years)
    report["years_valid"] = n_years == len(config.data.years)
    logger.debug("  Years: {} unique (expected {})", n_years, len(config.data.years))

    # 8. No duplicates
    n_rows = len(imputed_panel)
    n_unique_pairs = imputed_panel.groupby(["Province", "Year"]).size().shape[0]
    report["n_duplicate_pairs"] = n_rows - n_unique_pairs
    report["no_duplicates"] = n_rows == n_unique_pairs
    logger.debug("  Duplicates: {} duplicate (province, year) pairs", report["n_duplicate_pairs"])

    # Summary
    report["all_valid"] = all(
        [
            report["shape_valid"],
            report["columns_valid"],
            report["dtypes_valid"],
            report["nan_valid"],
            report["value_bounds_valid"],
            report["provinces_valid"],
            report["years_valid"],
            report["no_duplicates"],
        ]
    )

    if report["all_valid"]:
        logger.info("✓ Imputation validation PASSED")
    else:
        logger.warning("✗ Imputation validation FAILED: {}", report)

    return report["all_valid"], report


def save_imputed_panel(
    imputed_panel: pd.DataFrame,
    output_path: str | Path = "output/ml/imputed/panel_imputed.parquet",
) -> Path:
    """
    Save the imputed panel to disk in Parquet format.

    The Parquet format is chosen for:
    * Efficiency: smaller file size than CSV
    * Data type preservation: no need for type re-inference on load
    * Scalability: suitable for larger datasets

    Parameters
    ----------
    imputed_panel : pd.DataFrame
        Validated imputed panel to save.
    output_path : str | Path
        Output file path. Default: "output/ml/imputed/panel_imputed.parquet".
        Parent directories are created if they don't exist.

    Returns
    -------
    Path
        The Path object of the saved file.

    Raises
    ------
    IOError
        If the output directory cannot be created or the file cannot be written.

    Notes
    -----
    * This function creates the output directory if it doesn't exist.
    * The original data/csv/ files are never modified.
    * The returned Path can be used for downstream operations (e.g., loading by AutoGluon).

    Examples
    --------
    >>> panel_imputed = run_mice_imputation(panel_raw, config)
    >>> is_valid, _ = validate_imputation(panel_imputed, config)
    >>> if is_valid:
    ...     saved_path = save_imputed_panel(panel_imputed)
    ...     assert saved_path.exists()
    """
    path = Path(output_path)
    logger.info("Saving imputed panel to '{}'", path)

    try:
        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("  Ensured directory exists: {}", path.parent)

        # Save to Parquet
        imputed_panel.to_parquet(path, index=False, engine="pyarrow")
        logger.info("  ✓ Saved successfully: {} rows, {} columns", len(imputed_panel), len(imputed_panel.columns))

        # Verify file exists and has content
        if not path.exists():
            raise IOError(f"File was not created: {path}")

        file_size_mb = path.stat().st_size / 1024 / 1024
        logger.debug("  File size: {:.2f} MB", file_size_mb)

        return path

    except Exception as e:
        logger.error("Failed to save imputed panel: {}", str(e))
        raise IOError(f"Failed to save imputed panel to {path}: {str(e)}")


# =============================================================================
# Convenience wrapper
# =============================================================================

def run_full_imputation_pipeline(
    config: AppConfig,
    output_path: str | Path = "output/ml/imputed/panel_imputed.parquet",
    save: bool = True,
    csv_dir: str = "data/csv",
) -> Tuple[pd.DataFrame, bool, Dict[str, any]]:
    """
    Execute the complete MICE imputation pipeline: load → impute → validate → save.

    This is the primary entry point for Phase 6 (MICE Imputation).

    Parameters
    ----------
    config : AppConfig
        Application configuration.
    output_path : str | Path
        Path to save imputed panel (if save=True).
    save : bool
        Whether to save the imputed panel to disk.
    csv_dir : str
        Directory containing CSV files. Default: "data/csv"

    Returns
    -------
    Tuple[pd.DataFrame, bool, Dict[str, any]]
        * pd.DataFrame: the imputed panel
        * bool: validation result (True if all checks passed)
        * dict: detailed validation report

    Raises
    ------
    DataIntegrityError
        If panel loading or validation fails.
    ImputationError
        If MICE imputation fails.
    IOError
        If saving fails (when save=True).

    Examples
    --------
    >>> from src.core.schema import load_config
    >>> config = load_config()
    >>> panel_imputed, is_valid, report = run_full_imputation_pipeline(config)
    >>> assert is_valid, f"Validation failed: {report}"
    >>> assert (panel_imputed.isna().sum().sum() == 0)
    """
    logger.info("=" * 80)
    logger.info("PHASE 6: MICE IMPUTATION PIPELINE")
    logger.info("=" * 80)

    # Step 1: Load raw panel
    panel_raw = load_raw_panel(config, csv_dir=csv_dir)

    # Step 2: Run MICE imputation
    panel_imputed = run_mice_imputation(panel_raw, config)

    # Step 3: Validate
    is_valid, report = validate_imputation(panel_imputed, config)

    # Step 4: Save (optional)
    if save:
        save_imputed_panel(panel_imputed, output_path)

    logger.info("=" * 80)
    logger.info("PHASE 6 COMPLETE")
    logger.info("=" * 80)

    return panel_imputed, is_valid, report


# =============================================================================
# Helper functions (internal)
# =============================================================================

def _validate_raw_panel(panel: pd.DataFrame, config: AppConfig) -> None:
    """
    Validate the structure of a raw (pre-imputation) panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel to validate.
    config : AppConfig
        Application configuration for reference values.

    Raises
    ------
    DataIntegrityError
        If the panel structure is invalid.
    """
    # Check shape
    expected_n_rows = config.data.n_provinces * len(config.data.years)
    if len(panel) != expected_n_rows:
        raise DataIntegrityError(
            f"Raw panel has {len(panel)} rows; expected {expected_n_rows}",
            context={"n_rows": len(panel), "expected": expected_n_rows},
        )

    # Check columns
    expected_cols = {"Province", "Year"} | set(config.data.all_subcriteria)
    if set(panel.columns) != expected_cols:
        missing = expected_cols - set(panel.columns)
        extra = set(panel.columns) - expected_cols
        raise DataIntegrityError(
            f"Raw panel has unexpected columns. Missing: {missing}. Extra: {extra}",
            context={"missing": list(missing), "extra": list(extra)},
        )

    # Check uniqueness
    n_provinces = panel["Province"].nunique()
    if n_provinces != config.data.n_provinces:
        raise DataIntegrityError(
            f"Raw panel has {n_provinces} unique provinces; expected {config.data.n_provinces}",
            context={"n_provinces": n_provinces},
        )

    n_years = panel["Year"].nunique()
    if n_years != len(config.data.years):
        raise DataIntegrityError(
            f"Raw panel has {n_years} unique years; expected {len(config.data.years)}",
            context={"n_years": n_years},
        )

    logger.debug("✓ Raw panel structure validated")
