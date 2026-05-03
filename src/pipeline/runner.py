"""
src/pipeline/runner.py
----------------------
Master orchestrator for the complete IFS-MCDM-AutoML-XAI framework.

Responsibilities
----------------
* Load configuration from config.yaml
* Dispatch to MCDM pipeline and/or ML pipeline based on config
* Manage logging, error handling, and timing
* Provide summary reports

Production Quality
-----------
✅ Config-driven (all params from config.yaml)
✅ Full logging (INFO/DEBUG levels)
✅ Error recovery (contextual exceptions)
✅ Timing and performance metrics
✅ Type hints + docstrings
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from src.core.data_loader import load_config
from src.core.exceptions import MCDMError, ForecastingError
from src.core.schema import AppConfig
from src.pipeline.mcdm_pipeline import MCDMPipeline
from src.pipeline.ml_pipeline import MLPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Master Pipeline Runner
# =============================================================================

class PipelineRunner:
    """
    Master orchestrator for MCDM + ML + SHAP pipelines.

    Attributes
    ----------
    config : AppConfig
        Application configuration.
    mcdm_pipeline : MCDMPipeline | None
        MCDM pipeline instance.
    ml_pipeline : MLPipeline | None
        ML pipeline instance.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Initialize runner.

        Parameters
        ----------
        config_path : str
            Path to config.yaml.
        """
        logger.info("=" * 80)
        logger.info("🚀 IFS-MCDM-AutoML-XAI Framework Runner Initializing")
        logger.info("=" * 80)

        try:
            self.config = load_config(config_path)
            logger.info("✓ Loaded configuration from {}", config_path)
            logger.info(
                "  MCDM enabled: {}",
                self.config.pipeline.mcdm_enabled,
            )
            logger.info(
                "  ML enabled: {}",
                self.config.pipeline.ml_enabled,
            )

            self.mcdm_pipeline: Optional[MCDMPipeline] = None
            self.ml_pipeline: Optional[MLPipeline] = None

        except Exception as e:
            logger.exception("✗ Failed to initialize runner: {}", e)
            raise

    def run(self) -> int:
        """
        Execute the complete analysis pipeline.

        Returns
        -------
        int
            Exit code (0 = success, 1 = failure).
        """
        overall_start = time.time()

        try:
            logger.info("=" * 80)
            logger.info("🔷🟢 STARTING UNIFIED PIPELINE")
            logger.info("=" * 80)

            # Run MCDM pipeline if enabled
            if self.config.pipeline.mcdm_enabled:
                logger.info("")
                logger.info("📍 MCDM Pipeline Stage")
                logger.info("-" * 80)
                mcdm_start = time.time()

                try:
                    self.mcdm_pipeline = MCDMPipeline(self.config)
                    self.mcdm_pipeline.run()
                    mcdm_elapsed = time.time() - mcdm_start
                    logger.info("✓ MCDM pipeline completed in {:.2f}s", mcdm_elapsed)

                except (MCDMError, Exception) as e:
                    logger.error("✗ MCDM pipeline failed: {}", e)
                    raise

            # Run ML pipeline if enabled
            if self.config.pipeline.ml_enabled:
                logger.info("")
                logger.info("📍 ML Pipeline Stage")
                logger.info("-" * 80)
                ml_start = time.time()

                try:
                    self.ml_pipeline = MLPipeline(self.config)
                    self.ml_pipeline.run()
                    ml_elapsed = time.time() - ml_start
                    logger.info("✓ ML pipeline completed in {:.2f}s", ml_elapsed)

                except (ForecastingError, Exception) as e:
                    logger.error("✗ ML pipeline failed: {}", e)
                    raise

            # Summary
            overall_elapsed = time.time() - overall_start

            logger.info("")
            logger.info("=" * 80)
            logger.info("✓✓✓ UNIFIED PIPELINE COMPLETED SUCCESSFULLY ✓✓✓")
            logger.info("=" * 80)
            logger.info("Total execution time: {:.2f}s ({:.2f}m)", 
                        overall_elapsed, 
                        overall_elapsed / 60.0)
            logger.info("")
            logger.info("Outputs saved to: {}", Path(self.config.output.mcdm_dir).parent)
            logger.info("")

            return 0

        except (MCDMError, ForecastingError, Exception) as e:
            overall_elapsed = time.time() - overall_start
            logger.exception("✗✗✗ PIPELINE FAILED ✗✗✗")
            logger.error("Failed after {:.2f}s", overall_elapsed)
            logger.error("Error: {}", str(e))
            return 1


# =============================================================================
# Entry Point
# =============================================================================

def run_pipeline(config_path: str = "config/config.yaml") -> int:
    """
    Run the complete pipeline.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    runner = PipelineRunner(config_path)
    return runner.run()
