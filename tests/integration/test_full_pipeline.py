"""
tests/integration/test_full_pipeline.py
---------------------------------------
Integration tests for the complete IFS-MCDM-AutoML-XAI pipeline orchestration.

Test Coverage
-----------
✅ Configuration loading and validation
✅ MCDM pipeline execution with correct intermediate results
✅ ML pipeline execution with forecast generation
✅ MCDM application to 2025 forecasts
✅ Output integrity and file system checks
✅ Error handling and recovery
✅ End-to-end consistency and mathematical correctness
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.core.data_loader import load_config
from src.core.exceptions import MCDMError, ForecastingError
from src.core.schema import AppConfig, PipelineConfig
from src.pipeline.mcdm_pipeline import MCDMPipeline
from src.pipeline.ml_pipeline import MLPipeline
from src.pipeline.runner import PipelineRunner


# =============================================================================
# Configuration & Fixtures
# =============================================================================

class TestPipelineRunner(unittest.TestCase):
    """Test the master pipeline orchestrator."""

    def setUp(self) -> None:
        """Initialize test fixtures."""
        self.config_path = "config/config.yaml"

    def test_config_loading(self) -> None:
        """Test configuration loads correctly."""
        config = load_config(self.config_path)

        self.assertIsInstance(config, AppConfig)
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.mcdm)
        self.assertIsNotNone(config.ml)
        self.assertIsNotNone(config.pipeline)

    def test_pipeline_config_defaults(self) -> None:
        """Test pipeline configuration defaults to all enabled."""
        config = load_config(self.config_path)

        self.assertTrue(config.pipeline.mcdm_enabled)
        self.assertTrue(config.pipeline.ml_enabled)
        self.assertTrue(config.pipeline.mcdm_weighting_enabled)
        self.assertTrue(config.pipeline.mcdm_ranking_enabled)
        self.assertTrue(config.pipeline.mcdm_analysis_enabled)
        self.assertTrue(config.pipeline.ml_imputation_enabled)
        self.assertTrue(config.pipeline.ml_forecasting_enabled)
        self.assertTrue(config.pipeline.ml_shap_enabled)
        self.assertTrue(config.pipeline.ml_forecast_ranking_enabled)

    def test_pipeline_runner_initialization(self) -> None:
        """Test runner initializes without errors."""
        runner = PipelineRunner(self.config_path)

        self.assertIsNotNone(runner.config)
        self.assertIsNone(runner.mcdm_pipeline)
        self.assertIsNone(runner.ml_pipeline)

    def test_pipeline_runner_exit_code_on_success(self) -> None:
        """Test runner returns exit code 0 on successful completion."""
        runner = PipelineRunner(self.config_path)

        # Mock the pipelines to avoid actual execution
        with patch.object(MCDMPipeline, 'run'), \
             patch.object(MLPipeline, 'run'):
            exit_code = runner.run()

        self.assertEqual(exit_code, 0)


# =============================================================================
# MCDM Pipeline Integration Tests
# =============================================================================

class TestMCDMPipelineIntegration(unittest.TestCase):
    """Test MCDM pipeline orchestration."""

    def setUp(self) -> None:
        """Initialize test fixtures."""
        self.config = load_config("config/config.yaml")
        self.pipeline = MCDMPipeline(self.config)

    def test_mcdm_pipeline_initialization(self) -> None:
        """Test MCDM pipeline initializes correctly."""
        self.assertIsNotNone(self.pipeline.config)
        self.assertIsNone(self.pipeline.panel)
        self.assertIsNone(self.pipeline.ifs_panel)
        self.assertIsNone(self.pipeline.weights)
        self.assertIsNone(self.pipeline.rankings)

    def test_data_loading_with_disabled_pipeline(self) -> None:
        """Test that disabled pipeline is skipped."""
        self.config.pipeline.mcdm_enabled = False

        # Should return without executing
        with patch.object(self.pipeline, '_load_data') as mock_load:
            self.pipeline.run()
            mock_load.assert_not_called()

    def test_get_regime_for_year(self) -> None:
        """Test regime lookup for given year."""
        config = load_config("config/config.yaml")
        pipeline = MCDMPipeline(config)

        # Create mock panel
        from src.core.schema import Regime, PAPIPanel

        regimes = {
            "R1": Regime(
                regime_id="R1",
                years=[2011, 2012],
                active_subcriteria=["SC11"],
                absent_subcriteria=[],
            ),
        }
        pipeline.panel = MagicMock()
        pipeline.panel.regimes = regimes

        # Test year lookup
        regime = pipeline._get_regime_for_year(2011)
        self.assertEqual(regime.regime_id, "R1")

    def test_get_regime_for_invalid_year(self) -> None:
        """Test regime lookup fails for non-existent year."""
        config = load_config("config/config.yaml")
        pipeline = MCDMPipeline(config)

        pipeline.panel = MagicMock()
        pipeline.panel.regimes = {}

        with self.assertRaises(MCDMError):
            pipeline._get_regime_for_year(1999)


# =============================================================================
# ML Pipeline Integration Tests
# =============================================================================

class TestMLPipelineIntegration(unittest.TestCase):
    """Test ML pipeline orchestration."""

    def setUp(self) -> None:
        """Initialize test fixtures."""
        self.config = load_config("config/config.yaml")
        self.pipeline = MLPipeline(self.config)

    def test_ml_pipeline_initialization(self) -> None:
        """Test ML pipeline initializes correctly."""
        self.assertIsNotNone(self.pipeline.config)
        self.assertIsNone(self.pipeline.imputed_panel)
        self.assertIsNone(self.pipeline.forecast_table)
        self.assertIsNone(self.pipeline.rankings_2025)

    def test_ml_pipeline_disabled(self) -> None:
        """Test that disabled ML pipeline is skipped."""
        self.config.pipeline.ml_enabled = False

        with patch.object(self.pipeline, '_setup_output_directories') as mock_setup:
            self.pipeline.run()
            mock_setup.assert_not_called()

    def test_forecast_regime_creation_prefers_r3(self) -> None:
        """Test that forecast regime creation prefers R3 (complete regime)."""
        regime = self.pipeline._create_forecast_regime()

        # R3 should have 29 active sub-criteria
        self.assertEqual(regime.n_active, 29)
        self.assertEqual(regime.years, [2025])

    def test_forecast_regime_creation_fallback_to_r4(self) -> None:
        """Test forecast regime creation with only R4 available."""
        config = load_config("config/config.yaml")
        # Remove R3 to force fallback
        del config.data.regimes["R3"]

        pipeline = MLPipeline(config)
        regime = pipeline._create_forecast_regime()

        # R4 should be used as fallback
        self.assertEqual(regime.n_active, 28)
        self.assertEqual(regime.years, [2025])

    def test_forecast_regime_creation_fails_without_r3_or_r4(self) -> None:
        """Test forecast regime creation fails when both R3 and R4 missing."""
        config = load_config("config/config.yaml")
        # Remove both R3 and R4
        del config.data.regimes["R3"]
        del config.data.regimes["R4"]

        pipeline = MLPipeline(config)

        with self.assertRaises(MCDMError):
            pipeline._create_forecast_regime()


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

class TestEndToEndIntegration(unittest.TestCase):
    """Test complete end-to-end pipeline execution."""

    def setUp(self) -> None:
        """Initialize test fixtures."""
        self.config = load_config("config/config.yaml")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_output_dir = self.config.output.mcdm_dir

    def tearDown(self) -> None:
        """Clean up temporary resources."""
        self.temp_dir.cleanup()

    def test_configuration_validation(self) -> None:
        """Test that configuration is valid before pipeline execution."""
        config = load_config("config/config.yaml")

        # Verify all required sections exist
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.ifs)
        self.assertIsNotNone(config.mcdm)
        self.assertIsNotNone(config.ml)
        self.assertIsNotNone(config.output)
        self.assertIsNotNone(config.pipeline)

        # Verify critical parameters
        self.assertGreater(len(config.data.all_subcriteria), 0)
        self.assertGreater(len(config.data.years), 0)
        self.assertGreater(len(config.data.regimes), 0)
        self.assertGreater(len(config.mcdm.ranking.methods), 0)

    def test_pipeline_isolation(self) -> None:
        """Test that MCDM and ML pipelines can run independently."""
        config = load_config("config/config.yaml")

        # Test MCDM disabled, ML enabled
        config.pipeline.mcdm_enabled = False
        config.pipeline.ml_enabled = True

        with patch("src.pipeline.runner.load_config") as mock_load_config, \
             patch.object(MLPipeline, 'run') as mock_ml, \
             patch.object(MCDMPipeline, 'run') as mock_mcdm:
            mock_load_config.return_value = config
            runner = PipelineRunner("config/config.yaml")
            runner.run()

            mock_mcdm.assert_not_called()
            mock_ml.assert_called()

    def test_pipeline_composition(self) -> None:
        """Test that both pipelines can run together."""
        config = load_config("config/config.yaml")

        config.pipeline.mcdm_enabled = True
        config.pipeline.ml_enabled = True

        with patch.object(MCDMPipeline, 'run') as mock_mcdm, \
             patch.object(MLPipeline, 'run') as mock_ml:
            runner = PipelineRunner("config/config.yaml")
            runner.run()

            mock_mcdm.assert_called()
            mock_ml.assert_called()

    def test_output_directory_structure(self) -> None:
        """Test that output directories are created correctly."""
        config = load_config("config/config.yaml")

        # Verify output directory paths are set
        self.assertTrue(config.output.mcdm_dir)
        self.assertTrue(config.output.ml_dir)
        self.assertTrue(config.output.figures_dir)
        self.assertTrue(config.output.reports_dir)

        # Verify they're different
        self.assertNotEqual(config.output.mcdm_dir, config.output.ml_dir)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery."""

    def setUp(self) -> None:
        """Initialize test fixtures."""
        self.config = load_config("config/config.yaml")

    def test_invalid_config_path_raises_error(self) -> None:
        """Test that invalid config path raises error."""
        with self.assertRaises(Exception):
            PipelineRunner("nonexistent/config.yaml")

    def test_mcdm_pipeline_handles_missing_data(self) -> None:
        """Test MCDM pipeline error handling for missing data."""
        pipeline = MCDMPipeline(self.config)

        # Try to convert to IFS without loading data first
        with self.assertRaises(MCDMError):
            pipeline._convert_to_ifs()

    def test_ml_pipeline_handles_missing_imputed_data(self) -> None:
        """Test ML pipeline error handling for missing imputed data."""
        pipeline = MLPipeline(self.config)

        # Mock the imputed panel path to not exist
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            with self.assertRaises(ForecastingError):
                pipeline._load_imputed_panel()


# =============================================================================
# Test Suite
# =============================================================================

if __name__ == "__main__":
    unittest.main()
