"""
src/pipeline/ml_pipeline.py
----------------------------
ML pipeline orchestrator for MICE imputation + AutoGluon forecasting + SHAP explainability.

Responsibilities
----------------
* Load imputed panel from Phase 6 (output/ml/imputed/panel_imputed.parquet).
* Train AutoGluon TimeSeriesPredictor models for all 28-29 sub-criteria targets.
* Generate 2025 forecasts.
* Run SHAP explainability analysis on trained models.
* Save all outputs: models, forecasts, SHAP values, visualizations.
* Maintain data integrity and prevent leakage throughout.

Pipeline Flow
-----------
1. Validate configuration and output directories
2. Load MICE-imputed panel from Phase 6
3. Build TimeSeriesDataFrame objects (28-29 targets)
4. Train TimeSeriesPredictor models (28-29 models, sequential)
5. Generate 2025 forecasts (all targets)
6. Aggregate forecasts into consolidated table
7. Compute SHAP values (28-29 explainers)
8. Aggregate SHAP into global importance scores
9. Generate SHAP visualizations
10. Save all artifacts with metadata

Production Quality
-----------
✅ Config-driven (all params from config.yaml)
✅ Full logging (INFO/DEBUG levels)
✅ Error recovery (contextual exceptions)
✅ Data validation (shapes, NaN, bounds)
✅ Reproducibility (fixed random seeds)
✅ Type hints + docstrings
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from src.core.exceptions import ForecastingError
from src.core.schema import AppConfig
from src.ml.explainability.shap_explainer import (
    aggregate_shap_results,
    plot_all_shap_visualizations,
    run_shap_for_all_targets,
    save_shap_values,
)
from src.ml.forecasting.autogluon_forecaster import (
    aggregate_forecasts,
    build_timeseries_dataframes,
    forecast_all_targets,
    load_imputed_panel,
    save_forecast_output,
    train_predictors,
    validate_forecast_output,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# ML Pipeline Orchestrator
# =============================================================================

class MLPipeline:
    """
    Orchestrates the complete ML pipeline: MICE imputation + AutoGluon + SHAP.

    Attributes
    ----------
    config : AppConfig
        Application configuration with ML parameters.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize ML pipeline.

        Parameters
        ----------
        config : AppConfig
            Application configuration.
        """
        self.config = config
        logger.info("MLPipeline initialized")

    def run(self) -> None:
        """
        Execute the complete ML pipeline.

        Flow
        ----
        1. Setup & validation
        2. Load imputed panel
        3. Build & train models
        4. Generate forecasts
        5. Run SHAP analysis
        6. Visualize & save results

        Raises
        ------
        ForecastingError
            If any pipeline stage fails.
        """
        logger.info("=" * 80)
        logger.info("PHASE: ML PIPELINE (Imputation + Forecasting + SHAP)")
        logger.info("=" * 80)

        try:
            # Stage 1: Setup & Validation
            logger.info("\n[Stage 1/6] Setup & Validation")
            self._setup_output_directories()

            # Stage 2: Load Imputed Panel
            logger.info("\n[Stage 2/6] Load Imputed Panel (Phase 6 output)")
            imputed_panel = load_imputed_panel(
                imputed_path=f"{self.config.output.ml_dir}/imputed/panel_imputed.parquet"
            )

            # Stage 3: Build & Train Models
            logger.info("\n[Stage 3/6] Build TimeSeriesDataFrames & Train Models")
            ts_dfs = build_timeseries_dataframes(imputed_panel, self.config)
            predictors = train_predictors(ts_dfs, self.config)

            # Stage 4: Generate Forecasts
            logger.info("\n[Stage 4/6] Generate 2025 Forecasts")
            forecasts = forecast_all_targets(
                predictors,
                target_year=self.config.ml.forecasting.target_year,
            )
            forecast_table = aggregate_forecasts(forecasts, self.config)
            validate_forecast_output(forecast_table, self.config)

            # Save forecasts
            forecast_output_path = Path(self.config.output.ml_dir) / "forecasts" / "forecast_2025"
            save_forecast_output(
                forecast_table,
                output_path=forecast_output_path,
                file_format=self.config.output.tabular_format.value,
            )

            # Stage 5: Run SHAP Analysis
            logger.info("\n[Stage 5/6] Run SHAP Explainability Analysis")
            shap_results = run_shap_for_all_targets(
                predictors,
                imputed_panel,
                self.config,
            )

            # Aggregate & save SHAP values
            shap_aggregation = aggregate_shap_results(shap_results)
            shap_output_dir = Path(self.config.output.ml_dir) / "shap"
            save_shap_values(
                shap_results,
                output_dir=shap_output_dir,
                file_format="parquet",
            )
            logger.info("  ✓ SHAP values saved to '{}'", shap_output_dir)

            # Stage 6: Visualizations
            logger.info("\n[Stage 6/6] Generate SHAP Visualizations")
            figures_dir = Path(self.config.output.figures_dir) / "ml"
            plot_all_shap_visualizations(
                shap_results,
                output_figures_dir=figures_dir,
                top_n_features=15,
                top_n_provinces=5,
            )

            # Summary
            logger.info("\n" + "=" * 80)
            logger.info("ML PIPELINE COMPLETE ✓")
            logger.info("=" * 80)
            logger.info("\nOutputs:")
            logger.info("  Forecasts    : {}/forecast_2025.*", self.config.output.ml_dir)
            logger.info("  SHAP values  : {}/shap/", self.config.output.ml_dir)
            logger.info("  SHAP figures : {}/ml/", self.config.output.figures_dir)
            logger.info("  Models       : {}/ag_models/", self.config.output.ml_dir)

        except Exception as exc:
            logger.error("ML PIPELINE FAILED ✗")
            raise ForecastingError(
                f"ML pipeline execution failed: {exc}",
                context={"error": str(exc)},
            ) from exc

    def _setup_output_directories(self) -> None:
        """Create required output directories."""
        logger.info("Creating output directories")

        directories = [
            Path(self.config.output.ml_dir) / "imputed",
            Path(self.config.output.ml_dir) / "forecasts",
            Path(self.config.output.ml_dir) / "shap",
            Path(self.config.output.ml_dir) / "ag_models",
            Path(self.config.output.figures_dir) / "ml",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug("  Created: {}", directory)

        logger.info("  ✓ Output directories ready")

