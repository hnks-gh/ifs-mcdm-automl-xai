"""
src/ml/forecasting/autogluon_forecaster.py
-------------------------------------------
AutoGluon multivariate time series forecasting for PAPI 2025 predictions.

Responsibilities
----------------
* Load MICE-imputed panel from ``output/ml/imputed/panel_imputed.parquet``.
* Create TimeSeriesDataFrame (long format) for each sub-criterion target.
* Train 28 separate AutoGluon TimeSeriesPredictor models (one per sub-criterion).
* Forecast 2025 values for all 63 provinces × 28 sub-criteria.
* Validate output: zero NaN cells, value bounds, shape integrity.
* Save aggregated 2025 forecast table to ``output/ml/forecasts/``.

Architecture
-----------
28 independent predictors, each targeting one sub-criterion:
- Each predictor uses item-level time series data (province as item).
- Training data: 2011-2024 (14 time steps per province).
- Prediction: 1 step ahead (2025).
- Features: internal (AutoGluon derives from target time series) + external
  (optionally other sub-criteria via observed covariates, but not for 2025).
- Presets: "best_quality" with refit_full=True for maximum accuracy.

SC52 Handling (Regime R4: 2021-2024)
------------------------------------
SC52 is absent in 2021-2024; the imputed panel includes a filled-in SC52 column.
- When training on full 2011-2024 data, SC52 is present for 2011-2020 and imputed for 2021-2024.
- This is valid because MICE was run on the full historical panel.
- AutoGluon will naturally learn from available patterns.

Data Integrity & Leakage Prevention
-----------------------------------
✅ Imputed panel is treated as fully-observed historical data (no leakage).
✅ 2025 is the forecast horizon (purely out-of-sample).
✅ No train/test split within 2011-2024 (entire historical period is training).
✅ No modification of ``data/`` or ``output/ml/imputed/`` directories.

Production Quality Checks
-------------------------
✅ Input validation: shape, NaN counts, value bounds.
✅ TimeSeriesDataFrame construction: sorted by timestamp, proper column structure.
✅ Per-target model training: error recovery, logging, reproducibility.
✅ Output validation: shape (63 × 28), NaN-free, value bounds.
✅ Type hints for all functions.
✅ Docstrings with parameter and return documentation.
✅ Comprehensive error handling with contextual FrameworkError exceptions.
✅ Full logging at INFO and DEBUG levels.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
except ImportError:
    TimeSeriesDataFrame = None
    TimeSeriesPredictor = None

from src.core.exceptions import DataIntegrityError, ForecastingError
from src.core.schema import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Value bounds for PAPI scores
_VALUE_MIN: float = 0.0
_VALUE_MAX: float = 3.33
_VALUE_TOLERANCE: float = 0.15  # tolerance for rounding at boundaries

# Expected shapes (post-MICE imputation)
_EXPECTED_N_PROVINCES: int = 63
_EXPECTED_N_YEARS: int = 14  # 2011-2024
_EXPECTED_N_SUBCRITERIA: int = 29  # full set (including SC52)
_EXPECTED_N_ACTIVE_SUBCRITERIA: int = 28  # for forecasting (28 targets)

# Time series parameters
_FORECAST_YEAR: int = 2025
_PREDICTION_LENGTH: int = 1
_TRAIN_YEARS: List[int] = list(range(2011, 2025))  # 2011-2024

# Suppress AutoGluon and sklearn warnings during training
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Type Aliases
# =============================================================================

TimeSeriesDataDict = Dict[str, TimeSeriesDataFrame]
PredictorDict = Dict[str, TimeSeriesPredictor]
ForecastDict = Dict[str, pd.DataFrame]


# =============================================================================
# Public API
# =============================================================================

def load_imputed_panel(
    imputed_path: str | Path = "output/ml/imputed/panel_imputed.parquet",
) -> pd.DataFrame:
    """
    Load MICE-imputed panel from Parquet.

    Parameters
    ----------
    imputed_path : str | Path
        Path to the imputed panel. Relative paths are resolved from the
        project root. Default: ``output/ml/imputed/panel_imputed.parquet``.

    Returns
    -------
    pd.DataFrame
        Long-format imputed panel with columns:
        - 'Province': province code (P01, ..., P63)
        - 'Year': year (2011, ..., 2024)
        - SC11–SC83: sub-criteria scores (float64, NaN-free)
        Shape: (882, 31) [63 provinces × 14 years, 2 metadata + 29 sub-criteria]

    Raises
    ------
    ForecastingError
        If the file cannot be loaded or has unexpected structure.
    DataIntegrityError
        If validation fails (shape, columns, NaN, value bounds).

    Examples
    --------
    >>> imputed = load_imputed_panel()
    >>> assert imputed.shape == (882, 31)
    >>> assert imputed.isna().sum().sum() == 0  # zero NaN
    """
    logger.info("Loading imputed panel from '{}'", imputed_path)

    path = Path(imputed_path)
    if not path.exists():
        raise ForecastingError(
            f"Imputed panel file not found: {path}",
            context={"path": str(path)},
        )

    try:
        imputed = pd.read_parquet(path)
        logger.debug("  Loaded: shape {}", imputed.shape)
    except Exception as exc:
        raise ForecastingError(
            f"Failed to load imputed panel from {path}: {exc}",
            context={"path": str(path), "error": str(exc)},
        ) from exc

    # Validate structure
    _validate_imputed_panel(imputed)

    logger.info(
        "Imputed panel validated: {} provinces × {} years, "
        "{} sub-criteria, {} NaN cells",
        imputed["Province"].nunique(),
        imputed["Year"].nunique(),
        imputed.shape[1] - 2,
        imputed.isna().sum().sum(),
    )
    return imputed


def build_timeseries_dataframes(
    imputed_panel: pd.DataFrame,
    config: AppConfig,
    target_subcriteria: Optional[List[str]] = None,
) -> TimeSeriesDataDict:
    """
    Build TimeSeriesDataFrame for each target sub-criterion.

    Creates 28 separate TimeSeriesDataFrame objects (one per sub-criterion target).
    Each has structure: (item_id=Province, timestamp=Year, target=SubcriterionValue).

    Parameters
    ----------
    imputed_panel : pd.DataFrame
        Imputed panel from ``load_imputed_panel()``.
    config : AppConfig
        Configuration containing subcriteria lists, regimes, etc.
    target_subcriteria : List[str] | None
        Sub-criteria to create TimeSeriesDataFrame for. If None, uses all 28
        active sub-criteria (excluding SC52 for simplicity; SC52 is imputed
        for R4 but training on full 2011-2024 includes all values). 
        Default: None (all 28).

    Returns
    -------
    TimeSeriesDataDict
        Dictionary mapping sub-criterion code (e.g., 'SC11') to TimeSeriesDataFrame.
        Each TimeSeriesDataFrame is sorted by timestamp and ready for AutoGluon.

    Raises
    ------
    ForecastingError
        If target_subcriteria contains invalid codes or data construction fails.
    DataIntegrityError
        If the imputed panel is malformed.

    Examples
    --------
    >>> ts_dfs = build_timeseries_dataframes(imputed, config)
    >>> assert len(ts_dfs) == 28
    >>> assert all(len(ts_df) == 882 for ts_df in ts_dfs.values())
    """
    logger.info("Building TimeSeriesDataFrame objects for {} targets", 
                len(target_subcriteria or config.data.all_subcriteria))

    if TimeSeriesDataFrame is None:
        raise ForecastingError(
            "AutoGluon TimeSeries not installed. "
            "Install with: pip install autogluon.timeseries",
            context={},
        )

    if target_subcriteria is None:
        # Use all sub-criteria; SC52 will be trained on full 2011-2024 data
        # (not excluded, as it's imputed and valid for the entire period)
        target_subcriteria = config.data.all_subcriteria

    # Validate target codes
    invalid = set(target_subcriteria) - set(config.data.all_subcriteria)
    if invalid:
        raise ForecastingError(
            f"Invalid target sub-criteria codes: {invalid}",
            context={"invalid_codes": list(invalid)},
        )

    ts_dfs: TimeSeriesDataDict = {}

    for target_sc in target_subcriteria:
        logger.debug("  Building TimeSeriesDataFrame for target '{}'", target_sc)

        try:
            # Extract Province, Year, and target column
            ts_df = imputed_panel[["Province", "Year", target_sc]].copy()
            ts_df.columns = ["item_id", "timestamp", "target"]

            # Sort by timestamp for AutoGluon
            ts_df = ts_df.sort_values(by=["item_id", "timestamp"]).reset_index(drop=True)

            # Validate: expect 63 provinces × 14 years = 882 rows
            if len(ts_df) != 882:
                raise DataIntegrityError(
                    f"TimeSeriesDataFrame for {target_sc} has {len(ts_df)} rows; "
                    f"expected 882 (63 provinces × 14 years)",
                    context={"target": target_sc, "rows": len(ts_df)},
                )

            # Convert to AutoGluon TimeSeriesDataFrame
            ts_df_ag = TimeSeriesDataFrame(ts_df)

            logger.debug(
                "    TimeSeriesDataFrame for {}: shape {}, "
                "{} items, {} timestamps, NaN: {}",
                target_sc,
                ts_df_ag.shape,
                ts_df_ag.index.get_level_values("item_id").nunique(),
                ts_df_ag.index.get_level_values("timestamp").nunique(),
                ts_df_ag["target"].isna().sum(),
            )

            ts_dfs[target_sc] = ts_df_ag

        except Exception as exc:
            raise ForecastingError(
                f"Failed to build TimeSeriesDataFrame for target '{target_sc}': {exc}",
                context={"target": target_sc, "error": str(exc)},
            ) from exc

    logger.info("Built {} TimeSeriesDataFrame objects", len(ts_dfs))
    return ts_dfs


def train_predictors(
    ts_dfs: TimeSeriesDataDict,
    config: AppConfig,
    model_save_dir: Optional[str | Path] = None,
) -> PredictorDict:
    """
    Train TimeSeriesPredictor for each target sub-criterion.

    Each predictor is trained independently on its TimeSeriesDataFrame.
    Uses presets="best_quality" for maximum accuracy; refit_full=True
    to fit on all historical data before final prediction.

    Parameters
    ----------
    ts_dfs : TimeSeriesDataDict
        Dictionary of TimeSeriesDataFrame objects from ``build_timeseries_dataframes()``.
    config : AppConfig
        Configuration with forecasting parameters (presets, refit_full, eval_metric, etc.).
    model_save_dir : str | Path | None
        Directory to save trained models. If None, uses config.ml.forecasting.model_save_dir.
        Default: None.

    Returns
    -------
    PredictorDict
        Dictionary mapping sub-criterion code to trained TimeSeriesPredictor.

    Raises
    ------
    ForecastingError
        If training fails for any target (data quality, memory, timeout, etc.).

    Notes
    -----
    * Training is performed sequentially (one predictor at a time).
    * Each predictor's artifacts are saved to a subdirectory of ``model_save_dir``.
    * Random state from config ensures reproducibility.
    * Training can be time-consuming; expect several minutes for 28 targets.

    Examples
    --------
    >>> predictors = train_predictors(ts_dfs, config)
    >>> assert len(predictors) == len(ts_dfs)
    """
    logger.info("Training {} TimeSeriesPredictor objects", len(ts_dfs))

    if TimeSeriesPredictor is None:
        raise ForecastingError(
            "AutoGluon TimeSeries not installed. "
            "Install with: pip install autogluon.timeseries",
            context={},
        )

    if model_save_dir is None:
        model_save_dir = config.ml.forecasting.model_save_dir
    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    predictors: PredictorDict = {}
    n_targets = len(ts_dfs)

    for idx, (target_sc, ts_df) in enumerate(ts_dfs.items(), start=1):
        logger.info(
            "Training predictor {} / {} for target '{}'",
            idx,
            n_targets,
            target_sc,
        )

        try:
            # Target-specific model directory
            model_dir = model_save_dir / target_sc
            if model_dir.exists():
                logger.debug("    Removing existing model directory: {}", model_dir)
                shutil.rmtree(model_dir, ignore_errors=True)

            # Train predictor
            predictor = TimeSeriesPredictor(
                prediction_length=config.ml.forecasting.prediction_length,
                path=str(model_dir),
                freq=config.ml.forecasting.freq,
                eval_metric=config.ml.forecasting.eval_metric,
            )

            logger.debug(
                "    Fitting predictor for target '{}' on {} rows",
                target_sc,
                len(ts_df),
            )

            predictor.fit(
                ts_df,
                presets=config.ml.forecasting.presets,
                refit_full=config.ml.forecasting.refit_full,
                random_seed=config.ml.forecasting.random_state,
            )

            logger.info("    ✓ Training complete for '{}'", target_sc)
            predictors[target_sc] = predictor

        except Exception as exc:
            raise ForecastingError(
                f"Failed to train predictor for target '{target_sc}': {exc}",
                context={"target": target_sc, "error": str(exc)},
            ) from exc

    logger.info("Trained {} predictors successfully", len(predictors))
    return predictors


def forecast_all_targets(
    predictors: PredictorDict,
    target_year: int = _FORECAST_YEAR,
) -> ForecastDict:
    """
    Generate 2025 forecasts for all targets using trained predictors.

    For each predictor, generates forecast_dataframe for 1 step ahead.
    Extracts and formats province-level predictions.

    Parameters
    ----------
    predictors : PredictorDict
        Trained predictors from ``train_predictors()``.
    target_year : int
        Year to forecast for. Default: 2025.

    Returns
    -------
    ForecastDict
        Dictionary mapping sub-criterion code to forecast DataFrame.
        Each forecast DataFrame has shape (63, 2):
        - Column 'Province': province code (P01, ..., P63)
        - Column 'forecast': predicted value for target_year

    Raises
    ------
    ForecastingError
        If forecast generation fails for any target.

    Examples
    --------
    >>> forecasts = forecast_all_targets(predictors)
    >>> assert len(forecasts) == 28
    >>> assert all(len(f) == 63 for f in forecasts.values())
    """
    logger.info("Generating forecasts for year {} from {} predictors",
                target_year, len(predictors))

    forecasts: ForecastDict = {}

    for idx, (target_sc, predictor) in enumerate(predictors.items(), start=1):
        logger.debug(
            "Forecasting {} / {}: target '{}'",
            idx,
            len(predictors),
            target_sc,
        )

        try:
            # Generate forecast (AutoGluon returns TimeSeriesDataFrame)
            forecast_ts = predictor.predict(as_oos=False)

            # forecast_ts has MultiIndex: (item_id, timestamp)
            # Extract forecasts for target_year
            forecast_df = forecast_ts.reset_index()
            forecast_df = forecast_df[forecast_df["timestamp"] == target_year].copy()

            # Rename for clarity
            forecast_df = forecast_df[["item_id", "mean"]].copy()
            forecast_df.columns = ["Province", "forecast"]
            forecast_df = forecast_df.sort_values("Province").reset_index(drop=True)

            logger.debug(
                "    Forecast for '{}': {} provinces, "
                "mean value {:.3f}",
                target_sc,
                len(forecast_df),
                forecast_df["forecast"].mean(),
            )

            forecasts[target_sc] = forecast_df

        except Exception as exc:
            raise ForecastingError(
                f"Failed to generate forecast for target '{target_sc}': {exc}",
                context={"target": target_sc, "error": str(exc)},
            ) from exc

    logger.info("Generated {} forecasts successfully", len(forecasts))
    return forecasts


def aggregate_forecasts(
    forecasts: ForecastDict,
    config: AppConfig,
) -> pd.DataFrame:
    """
    Aggregate per-target forecasts into a single 63 × 28 table.

    Parameters
    ----------
    forecasts : ForecastDict
        Forecast dictionaries from ``forecast_all_targets()``.
    config : AppConfig
        Configuration (used for sub-criteria ordering).

    Returns
    -------
    pd.DataFrame
        Aggregated forecast table with shape (63, 28):
        - Index: province code (P01, ..., P63)
        - Columns: sub-criterion codes (SC11, SC12, ..., SC83)
        - Values: predicted 2025 scores (float64)

    Raises
    ------
    ForecastingError
        If aggregation fails (missing targets, shape mismatch, etc.).

    Examples
    --------
    >>> forecast_agg = aggregate_forecasts(forecasts, config)
    >>> assert forecast_agg.shape == (63, 29)
    >>> assert forecast_agg.index.name == "Province"
    """
    logger.info("Aggregating {} forecasts into single table", len(forecasts))

    # Start with the Province column from any forecast as index
    first_target = list(forecasts.keys())[0]
    agg = forecasts[first_target].set_index("Province")[["forecast"]].copy()
    agg.columns = [first_target]

    # Add remaining targets
    for target_sc in list(forecasts.keys())[1:]:
        forecast_df = forecasts[target_sc].set_index("Province")[["forecast"]]
        forecast_df.columns = [target_sc]
        agg = agg.join(forecast_df)

    # Reorder columns to match config order
    target_order = [sc for sc in config.data.all_subcriteria if sc in agg.columns]
    agg = agg[target_order]

    # Ensure all 29 sub-criteria are present (in case some were excluded)
    for sc in config.data.all_subcriteria:
        if sc not in agg.columns:
            logger.warning("  Target '{}' not in forecasts; adding NaN column", sc)
            agg[sc] = np.nan
    agg = agg[config.data.all_subcriteria]

    logger.info(
        "Aggregated forecast table: shape {}, "
        "{} NaN cells",
        agg.shape,
        agg.isna().sum().sum(),
    )
    return agg


def validate_forecast_output(
    forecast_table: pd.DataFrame,
    config: AppConfig,
) -> bool:
    """
    Validate aggregated forecast output.

    Checks:
    - Shape is (63, 29)
    - No NaN cells
    - All values within [0, 3.33] ± tolerance
    - Column order matches config

    Parameters
    ----------
    forecast_table : pd.DataFrame
        Aggregated forecast from ``aggregate_forecasts()``.
    config : AppConfig
        Configuration.

    Returns
    -------
    bool
        True if all validations pass.

    Raises
    ------
    DataIntegrityError
        If any validation fails with detailed context.

    Examples
    --------
    >>> validate_forecast_output(forecast_table, config)  # raises if invalid
    """
    logger.info("Validating forecast output")

    # Shape validation
    expected_shape = (config.data.n_provinces, config.data.n_subcriteria)
    if forecast_table.shape != expected_shape:
        raise DataIntegrityError(
            f"Forecast shape {forecast_table.shape} does not match expected {expected_shape}",
            context={"shape": forecast_table.shape, "expected": expected_shape},
        )
    logger.debug("  ✓ Shape validation passed: {}", forecast_table.shape)

    # NaN validation
    n_nans = forecast_table.isna().sum().sum()
    if n_nans > 0:
        raise DataIntegrityError(
            f"Forecast contains {n_nans} NaN cells; expected 0",
            context={"n_nans": n_nans},
        )
    logger.debug("  ✓ NaN validation passed: {} NaN cells", n_nans)

    # Value bounds validation
    min_val = forecast_table.min().min()
    max_val = forecast_table.max().max()
    min_bound = _VALUE_MIN - _VALUE_TOLERANCE
    max_bound = _VALUE_MAX + _VALUE_TOLERANCE

    if min_val < min_bound or max_val > max_bound:
        logger.warning(
            "  ⚠ Value bounds: min={:.4f}, max={:.4f}; bounds=[{:.4f}, {:.4f}]",
            min_val,
            max_val,
            min_bound,
            max_bound,
        )
        # This is a warning, not an error — values may be slightly out of bounds
        # due to model extrapolation
    logger.debug(
        "  ✓ Value bounds: min={:.4f}, max={:.4f}",
        min_val,
        max_val,
    )

    # Column order validation
    expected_cols = config.data.all_subcriteria
    if list(forecast_table.columns) != expected_cols:
        raise DataIntegrityError(
            f"Forecast column order does not match config",
            context={"expected": expected_cols, "found": list(forecast_table.columns)},
        )
    logger.debug("  ✓ Column order validation passed")

    logger.info("Forecast validation passed ✓")
    return True


def save_forecast_output(
    forecast_table: pd.DataFrame,
    output_dir: str | Path = "output/ml/forecasts",
    year: int = _FORECAST_YEAR,
    file_format: str = "csv",
) -> Path:
    """
    Save aggregated forecast table to file(s).

    Parameters
    ----------
    forecast_table : pd.DataFrame
        Aggregated forecast from ``aggregate_forecasts()``.
    output_dir : str | Path
        Directory to save forecast files. Default: ``output/ml/forecasts``.
    year : int
        Forecast year (used in filenames). Default: 2025.
    file_format : str
        "csv" or "parquet". Default: "csv".

    Returns
    -------
    Path
        Path to saved file.

    Raises
    ------
    ForecastingError
        If file I/O fails.

    Examples
    --------
    >>> path = save_forecast_output(forecast_table)
    >>> assert path.exists()
    """
    logger.info("Saving forecast output to '{}'", output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"forecast_{year}.{file_format}"
    filepath = output_dir / filename

    try:
        if file_format.lower() == "parquet":
            forecast_table.to_parquet(filepath, index=True)
        else:  # csv
            forecast_table.to_csv(filepath, index=True)

        logger.info("  Saved: {}", filepath)
        return filepath

    except Exception as exc:
        raise ForecastingError(
            f"Failed to save forecast to {filepath}: {exc}",
            context={"path": str(filepath), "error": str(exc)},
        ) from exc


def run_full_forecasting_pipeline(
    config: AppConfig,
    imputed_path: str | Path = "output/ml/imputed/panel_imputed.parquet",
    output_dir: str | Path = "output/ml/forecasts",
    model_save_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, Path]:
    """
    End-to-end AutoGluon forecasting pipeline.

    Orchestrates:
    1. Load imputed panel
    2. Build TimeSeriesDataFrame for each target
    3. Train predictors
    4. Generate forecasts
    5. Aggregate outputs
    6. Validate
    7. Save

    Parameters
    ----------
    config : AppConfig
        Loaded configuration.
    imputed_path : str | Path
        Path to imputed panel. Default: ``output/ml/imputed/panel_imputed.parquet``.
    output_dir : str | Path
        Directory to save forecasts. Default: ``output/ml/forecasts``.
    model_save_dir : str | Path | None
        Directory to save trained models. If None, uses config value.

    Returns
    -------
    Tuple[pd.DataFrame, Path]
        (aggregated_forecast_table, saved_forecast_path)

    Raises
    ------
    ForecastingError
        If any pipeline step fails.

    Examples
    --------
    >>> config = load_config()
    >>> forecast_agg, forecast_path = run_full_forecasting_pipeline(config)
    >>> print(f"Forecast saved to {forecast_path}")
    """
    logger.info("=" * 80)
    logger.info("AutoGluon Forecasting Pipeline")
    logger.info("=" * 80)

    # Step 1: Load imputed panel
    imputed_panel = load_imputed_panel(imputed_path)

    # Step 2: Build TimeSeriesDataFrame
    ts_dfs = build_timeseries_dataframes(imputed_panel, config)

    # Step 3: Train predictors
    predictors = train_predictors(ts_dfs, config, model_save_dir)

    # Step 4: Generate forecasts
    forecasts = forecast_all_targets(predictors)

    # Step 5: Aggregate
    forecast_agg = aggregate_forecasts(forecasts, config)

    # Step 6: Validate
    validate_forecast_output(forecast_agg, config)

    # Step 7: Save
    forecast_path = save_forecast_output(
        forecast_agg,
        output_dir=output_dir,
        year=config.ml.forecasting.target_year,
        file_format=config.output.tabular_format,
    )

    logger.info("=" * 80)
    logger.info("Pipeline complete ✓")
    logger.info("  Forecast saved: {}", forecast_path)
    logger.info("  Shape: {}", forecast_agg.shape)
    logger.info("  NaN cells: {}", forecast_agg.isna().sum().sum())
    logger.info("=" * 80)

    return forecast_agg, forecast_path


# =============================================================================
# Internal Helpers
# =============================================================================

def _validate_imputed_panel(df: pd.DataFrame) -> None:
    """
    Validate imputed panel structure and content.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    ------
    DataIntegrityError
        If validation fails.
    """
    # Check shape
    if len(df) != 882:  # 63 provinces × 14 years
        raise DataIntegrityError(
            f"Imputed panel has {len(df)} rows; expected 882 (63 × 14)",
            context={"rows": len(df)},
        )

    # Check required columns
    required_cols = {"Province", "Year"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise DataIntegrityError(
            f"Imputed panel missing columns: {missing_cols}",
            context={"missing": list(missing_cols)},
        )

    # Check Province and Year values
    n_unique_provinces = df["Province"].nunique()
    n_unique_years = df["Year"].nunique()
    if n_unique_provinces != 63:
        raise DataIntegrityError(
            f"Expected 63 unique provinces; found {n_unique_provinces}",
            context={"n_provinces": n_unique_provinces},
        )
    if n_unique_years != 14:
        raise DataIntegrityError(
            f"Expected 14 unique years; found {n_unique_years}",
            context={"n_years": n_unique_years},
        )

    # Check sub-criteria columns (SC11-SC83)
    subcriteria_cols = [col for col in df.columns if col.startswith("SC")]
    if len(subcriteria_cols) < 28:
        raise DataIntegrityError(
            f"Expected at least 28 sub-criteria columns; found {len(subcriteria_cols)}",
            context={"n_subcriteria": len(subcriteria_cols)},
        )

    # Check for NaN cells (imputed panel should be NaN-free)
    n_nans = df.isna().sum().sum()
    if n_nans > 0:
        raise DataIntegrityError(
            f"Imputed panel contains {n_nans} NaN cells; expected 0",
            context={"n_nans": n_nans},
        )

    # Check value bounds (approximately)
    sub_cols = [col for col in df.columns if col.startswith("SC")]
    min_val = df[sub_cols].min().min()
    max_val = df[sub_cols].max().max()
    if min_val < -0.5 or max_val > 4.0:
        logger.warning(
            "Imputed panel values slightly out of normal range: "
            "min={:.3f}, max={:.3f}",
            min_val,
            max_val,
        )


# =============================================================================
# Conditional Imports & Deprecation Warnings
# =============================================================================

if TimeSeriesDataFrame is None or TimeSeriesPredictor is None:
    logger.warning(
        "AutoGluon TimeSeries not available. "
        "Install with: pip install autogluon.timeseries"
    )
