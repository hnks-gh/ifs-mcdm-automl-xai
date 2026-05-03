"""
src/ml/forecasting.__init__.py
-------------------------------
Public API for AutoGluon time series forecasting module.
"""

from src.ml.forecasting.autogluon_forecaster import (
    aggregate_forecasts,
    build_timeseries_dataframes,
    forecast_all_targets,
    load_imputed_panel,
    run_full_forecasting_pipeline,
    save_forecast_output,
    train_predictors,
    validate_forecast_output,
)

__all__ = [
    "load_imputed_panel",
    "build_timeseries_dataframes",
    "train_predictors",
    "forecast_all_targets",
    "aggregate_forecasts",
    "validate_forecast_output",
    "save_forecast_output",
    "run_full_forecasting_pipeline",
]
