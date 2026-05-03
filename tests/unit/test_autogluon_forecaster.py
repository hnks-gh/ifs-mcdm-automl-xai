"""
tests/unit/test_autogluon_forecaster.py
----------------------------------------
Comprehensive unit tests for AutoGluon time series forecasting module.

Test Coverage
-------------
✓ load_imputed_panel: file I/O, validation, error handling
✓ build_timeseries_dataframes: TS data construction, validation, edge cases
✓ validate_forecast_output: shape, NaN, bounds checking
✓ aggregate_forecasts: merging per-target forecasts
✓ save_forecast_output: file I/O (CSV, Parquet)
✓ Integration: full pipeline orchestration (mocked AutoGluon)

Mocking Strategy
----------------
train_predictors and forecast_all_targets are integration-level functions that
depend on AutoGluon, which is computationally expensive. We mock these to enable
unit-level testing of data preparation and orchestration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import DataIntegrityError, ForecastingError
from src.core.data_loader import load_config
from src.ml.forecasting.autogluon_forecaster import (
    aggregate_forecasts,
    build_timeseries_dataframes,
    load_imputed_panel,
    save_forecast_output,
    validate_forecast_output,
)

# Check if AutoGluon is available
try:
    import autogluon.timeseries
    HAS_AUTOGLUON = True
except ImportError:
    HAS_AUTOGLUON = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Load test configuration."""
    return load_config("config/config.yaml")


@pytest.fixture
def synthetic_imputed_panel(config: any) -> pd.DataFrame:
    """Create a synthetic imputed panel for testing.
    
    Structure: 882 rows (63 provinces × 14 years), 31 columns
    (Province, Year, SC11-SC83)
    """
    n_provinces = config.data.n_provinces  # 63
    n_years = len(config.data.years)  # 14
    n_rows = n_provinces * n_years  # 882
    
    # Create synthetic data
    data = {
        "Province": [f"P{i:02d}" for i in range(1, n_provinces + 1)] * n_years,
        "Year": sorted([y for y in config.data.years for _ in range(n_provinces)]),
    }
    
    # Add sub-criteria columns with realistic values [0, 3.33]
    np.random.seed(42)
    for sc in config.data.all_subcriteria:
        # Generate semi-realistic values: some correlation with year
        base = np.random.uniform(1.0, 3.0, n_rows)
        trend = np.tile(np.linspace(0, 0.3, n_years), n_provinces)
        data[sc] = np.clip(base + trend, 0.0, 3.33)
    
    panel = pd.DataFrame(data)
    
    # Verify shape
    assert panel.shape == (882, 31), f"Expected (882, 31), got {panel.shape}"
    assert panel.isna().sum().sum() == 0, "Synthetic panel should have no NaN"
    
    return panel


@pytest.fixture
def synthetic_forecasts(config) -> dict:
    """Create synthetic forecast dictionaries for testing."""
    forecasts = {}
    for sc in config.data.all_subcriteria:
        forecast_df = pd.DataFrame({
            "Province": [f"P{i:02d}" for i in range(1, 64)],
            "forecast": np.random.uniform(1.0, 3.3, 63),
        })
        forecasts[sc] = forecast_df
    return forecasts


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Tests: load_imputed_panel
# =============================================================================

class TestLoadImputedPanel:
    """Tests for load_imputed_panel function."""
    
    def test_load_valid_parquet(self, synthetic_imputed_panel, temp_output_dir):
        """Test loading valid imputed panel from Parquet."""
        # Save synthetic panel
        parquet_path = temp_output_dir / "panel_imputed.parquet"
        synthetic_imputed_panel.to_parquet(parquet_path)
        
        # Load it
        loaded = load_imputed_panel(parquet_path)
        
        # Verify
        assert loaded.shape == synthetic_imputed_panel.shape
        assert list(loaded.columns) == list(synthetic_imputed_panel.columns)
        pd.testing.assert_frame_equal(loaded, synthetic_imputed_panel)
    
    def test_load_missing_file(self, temp_output_dir):
        """Test error handling for missing file."""
        missing_path = temp_output_dir / "nonexistent.parquet"
        
        with pytest.raises(ForecastingError, match="not found"):
            load_imputed_panel(missing_path)
    
    def test_validate_shape(self, synthetic_imputed_panel, temp_output_dir, config):
        """Test shape validation (882 rows, 31 cols)."""
        # Save synthetic panel
        parquet_path = temp_output_dir / "panel_imputed.parquet"
        synthetic_imputed_panel.to_parquet(parquet_path)
        
        # Load should succeed
        loaded = load_imputed_panel(parquet_path)
        assert loaded.shape == (882, 31)
    
    def test_validate_shape_failure_wrong_rows(self, temp_output_dir):
        """Test shape validation failure: wrong number of rows."""
        # Create invalid panel with wrong number of rows
        invalid_panel = pd.DataFrame({
            "Province": [f"P{i:02d}" for i in range(1, 50)],
            "Year": [2020] * 49,
        })
        parquet_path = temp_output_dir / "invalid.parquet"
        invalid_panel.to_parquet(parquet_path)
        
        with pytest.raises(DataIntegrityError, match="882"):
            load_imputed_panel(parquet_path)
    
    def test_validate_missing_province_column(self, temp_output_dir):
        """Test validation failure: missing Province column."""
        invalid_panel = pd.DataFrame({
            "Year": [2020] * 882,
            "SC11": np.random.uniform(0, 3.33, 882),
        })
        parquet_path = temp_output_dir / "invalid.parquet"
        invalid_panel.to_parquet(parquet_path)
        
        with pytest.raises(DataIntegrityError, match="Province"):
            load_imputed_panel(parquet_path)
    
    def test_validate_unique_provinces(self, temp_output_dir):
        """Test validation of unique province count."""
        # Create panel with wrong number of unique provinces (29 instead of 63)
        invalid_panel = pd.DataFrame({
            "Province": [f"P{i:02d}" for i in range(1, 30)] * 14,  # 29 unique provinces repeated 14 times
            "Year": sorted([y for y in range(2011, 2025) for _ in range(29)]),  # 14 years × 29 = 406 rows
            "SC11": np.random.uniform(0, 3.33, 29 * 14),
        })
        parquet_path = temp_output_dir / "invalid.parquet"
        invalid_panel.to_parquet(parquet_path)
        
        with pytest.raises(DataIntegrityError, match="63"):
            load_imputed_panel(parquet_path)
    
    def test_validate_unique_years(self, temp_output_dir):
        """Test validation of unique year count."""
        # Create panel with wrong number of years (10 instead of 14)
        invalid_panel = pd.DataFrame({
            "Province": [f"P{i:02d}" for i in range(1, 64)] * 10,  # 63 provinces repeated 10 times
            "Year": sorted([y for y in range(2011, 2021) for _ in range(63)]),  # 10 years × 63 = 630 rows
            "SC11": np.random.uniform(0, 3.33, 63 * 10),
        })
        parquet_path = temp_output_dir / "invalid.parquet"
        invalid_panel.to_parquet(parquet_path)
        
        with pytest.raises(DataIntegrityError, match="14"):
            load_imputed_panel(parquet_path)
    
    def test_validate_nan_cells(self, synthetic_imputed_panel, temp_output_dir):
        """Test validation failure: NaN cells in imputed panel."""
        # Create panel with NaN cells by copying the synthetic one and introducing NaN
        invalid_panel = synthetic_imputed_panel.copy()
        invalid_panel.loc[0, "SC11"] = np.nan  # Introduce NaN
        
        parquet_path = temp_output_dir / "invalid.parquet"
        invalid_panel.to_parquet(parquet_path)
        
        with pytest.raises(DataIntegrityError, match="NaN"):
            load_imputed_panel(parquet_path)


# =============================================================================
# Tests: build_timeseries_dataframes
# =============================================================================

@pytest.mark.skipif(not HAS_AUTOGLUON, reason="AutoGluon not installed")
class TestBuildTimeseriesDataframes:
    """Tests for build_timeseries_dataframes function."""
    
    def test_build_all_targets(self, synthetic_imputed_panel, config):
        """Test building TimeSeriesDataFrame for all targets."""
        # Build without specifying targets (defaults to all)
        ts_dfs = build_timeseries_dataframes(synthetic_imputed_panel, config)
        
        # Verify
        assert len(ts_dfs) == 29, f"Expected 29 TimeSeriesDataFrames, got {len(ts_dfs)}"
        for sc, ts_df in ts_dfs.items():
            assert sc in config.data.all_subcriteria
            # AutoGluon TimeSeriesDataFrame has MultiIndex (item_id, timestamp)
            assert len(ts_df) == 882, f"Expected 882 rows for {sc}, got {len(ts_df)}"
    
    def test_build_subset_targets(self, synthetic_imputed_panel, config):
        """Test building TimeSeriesDataFrame for specific targets."""
        targets = ["SC11", "SC12", "SC21"]
        ts_dfs = build_timeseries_dataframes(
            synthetic_imputed_panel,
            config,
            target_subcriteria=targets,
        )
        
        assert len(ts_dfs) == 3
        assert set(ts_dfs.keys()) == set(targets)
    
    def test_build_invalid_target_codes(self, synthetic_imputed_panel, config):
        """Test error handling for invalid target codes."""
        invalid_targets = ["SC11", "INVALID", "SC99"]
        
        with pytest.raises(ForecastingError, match="Invalid"):
            build_timeseries_dataframes(
                synthetic_imputed_panel,
                config,
                target_subcriteria=invalid_targets,
            )
    
    def test_ts_dataframe_structure(self, synthetic_imputed_panel, config):
        """Test internal structure of built TimeSeriesDataFrame."""
        ts_dfs = build_timeseries_dataframes(
            synthetic_imputed_panel,
            config,
            target_subcriteria=["SC11"],
        )
        
        ts_df = ts_dfs["SC11"]
        
        # Verify columns after conversion
        # AutoGluon TimeSeriesDataFrame should have "target" column
        assert "target" in ts_df.columns
        
        # Verify MultiIndex structure
        assert ts_df.index.nlevels == 2  # (item_id, timestamp)
        assert ts_df.index.names == ["item_id", "timestamp"]
    
    def test_ts_dataframe_no_nans(self, synthetic_imputed_panel, config):
        """Test that built TimeSeriesDataFrame has no NaN in target."""
        ts_dfs = build_timeseries_dataframes(synthetic_imputed_panel, config)
        
        for sc, ts_df in ts_dfs.items():
            n_nans = ts_df["target"].isna().sum()
            assert n_nans == 0, f"TimeSeriesDataFrame for {sc} has {n_nans} NaN values"


# =============================================================================
# Tests: validate_forecast_output
# =============================================================================

class TestValidateForecastOutput:
    """Tests for validate_forecast_output function."""
    
    def test_validate_valid_forecast(self, config):
        """Test validation of valid forecast output."""
        # Create valid forecast
        forecast = pd.DataFrame(
            np.random.uniform(1.0, 3.3, (63, 29)),
            index=[f"P{i:02d}" for i in range(1, 64)],
            columns=config.data.all_subcriteria,
        )
        forecast.index.name = "Province"
        
        # Should not raise
        result = validate_forecast_output(forecast, config)
        assert result is True
    
    def test_validate_shape_mismatch_rows(self, config):
        """Test validation failure: wrong number of provinces."""
        forecast = pd.DataFrame(
            np.random.uniform(1.0, 3.3, (50, 29)),
            index=[f"P{i:02d}" for i in range(1, 51)],
            columns=config.data.all_subcriteria,
        )
        forecast.index.name = "Province"
        
        with pytest.raises(DataIntegrityError, match="shape"):
            validate_forecast_output(forecast, config)
    
    def test_validate_shape_mismatch_cols(self, config):
        """Test validation failure: wrong number of columns."""
        forecast = pd.DataFrame(
            np.random.uniform(1.0, 3.3, (63, 25)),
            index=[f"P{i:02d}" for i in range(1, 64)],
            columns=config.data.all_subcriteria[:25],
        )
        forecast.index.name = "Province"
        
        with pytest.raises(DataIntegrityError, match="shape"):
            validate_forecast_output(forecast, config)
    
    def test_validate_nan_cells(self, config):
        """Test validation failure: NaN cells in forecast."""
        forecast = pd.DataFrame(
            np.random.uniform(1.0, 3.3, (63, 29)),
            index=[f"P{i:02d}" for i in range(1, 64)],
            columns=config.data.all_subcriteria,
        )
        forecast.index.name = "Province"
        forecast.iloc[0, 0] = np.nan  # Introduce NaN
        
        with pytest.raises(DataIntegrityError, match="NaN"):
            validate_forecast_output(forecast, config)
    
    def test_validate_value_bounds_out_of_range(self, config):
        """Test warning for values out of normal bounds."""
        forecast = pd.DataFrame(
            np.random.uniform(3.0, 5.0, (63, 29)),  # out of [0, 3.33]
            index=[f"P{i:02d}" for i in range(1, 64)],
            columns=config.data.all_subcriteria,
        )
        forecast.index.name = "Province"
        
        # Should log warning but not raise
        result = validate_forecast_output(forecast, config)
        assert result is True
    
    def test_validate_column_order_mismatch(self, config):
        """Test validation failure: wrong column order."""
        columns_reversed = list(reversed(config.data.all_subcriteria))
        forecast = pd.DataFrame(
            np.random.uniform(1.0, 3.3, (63, 29)),
            index=[f"P{i:02d}" for i in range(1, 64)],
            columns=columns_reversed,
        )
        forecast.index.name = "Province"
        
        with pytest.raises(DataIntegrityError, match="column"):
            validate_forecast_output(forecast, config)


# =============================================================================
# Tests: aggregate_forecasts
# =============================================================================

class TestAggregateForecasts:
    """Tests for aggregate_forecasts function."""
    
    def test_aggregate_all_forecasts(self, synthetic_forecasts, config):
        """Test aggregating all per-target forecasts."""
        agg = aggregate_forecasts(synthetic_forecasts, config)
        
        # Verify shape
        assert agg.shape == (63, 29)
        assert agg.index.name == "Province"
        
        # Verify columns
        assert list(agg.columns) == config.data.all_subcriteria
        
        # Verify no NaN
        assert agg.isna().sum().sum() == 0
    
    def test_aggregate_missing_target(self, synthetic_forecasts, config):
        """Test aggregation when a target is missing."""
        # Remove one forecast
        del synthetic_forecasts["SC11"]
        
        # Aggregate (should add NaN column for missing target)
        agg = aggregate_forecasts(synthetic_forecasts, config)
        
        # Shape should still be (63, 29)
        assert agg.shape == (63, 29)
        # SC11 column should be all NaN
        assert agg["SC11"].isna().all()
    
    def test_aggregate_forecast_values(self, config):
        """Test that aggregated forecast values match input forecasts."""
        # Create simple forecasts for testing
        forecasts = {}
        expected_values = {}
        for sc in ["SC11", "SC12", "SC21"]:
            df = pd.DataFrame({
                "Province": [f"P{i:02d}" for i in range(1, 64)],
                "forecast": np.arange(1.0, 64.0),  # 1.0 to 63.0
            })
            forecasts[sc] = df
            expected_values[sc] = np.arange(1.0, 64.0)
        
        agg = aggregate_forecasts(forecasts, config)
        
        # Verify values for tested sub-criteria
        for sc in ["SC11", "SC12", "SC21"]:
            agg_vals = agg[sc].values
            np.testing.assert_array_almost_equal(agg_vals, expected_values[sc])


# =============================================================================
# Tests: save_forecast_output
# =============================================================================

class TestSaveForecastOutput:
    """Tests for save_forecast_output function."""
    
    def test_save_csv(self, config, temp_output_dir):
        """Test saving forecast to CSV."""
        forecast = pd.DataFrame(
            np.random.uniform(1.0, 3.3, (63, 29)),
            index=[f"P{i:02d}" for i in range(1, 64)],
            columns=config.data.all_subcriteria,
        )
        forecast.index.name = "Province"
        
        path = save_forecast_output(
            forecast,
            output_dir=temp_output_dir,
            year=2025,
            file_format="csv",
        )
        
        # Verify file exists
        assert path.exists()
        assert path.suffix == ".csv"
        
        # Verify content
        loaded = pd.read_csv(path, index_col="Province")
        pd.testing.assert_frame_equal(loaded, forecast)
    
    def test_save_parquet(self, config, temp_output_dir):
        """Test saving forecast to Parquet."""
        forecast = pd.DataFrame(
            np.random.uniform(1.0, 3.3, (63, 29)),
            index=[f"P{i:02d}" for i in range(1, 64)],
            columns=config.data.all_subcriteria,
        )
        forecast.index.name = "Province"
        
        path = save_forecast_output(
            forecast,
            output_dir=temp_output_dir,
            year=2025,
            file_format="parquet",
        )
        
        # Verify file exists
        assert path.exists()
        assert path.suffix == ".parquet"
        
        # Verify content
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(loaded, forecast)
    
    def test_save_creates_output_dir(self, config, temp_output_dir):
        """Test that save_forecast_output creates output directory if missing."""
        forecast = pd.DataFrame(
            np.random.uniform(1.0, 3.3, (63, 29)),
            index=[f"P{i:02d}" for i in range(1, 64)],
            columns=config.data.all_subcriteria,
        )
        forecast.index.name = "Province"
        
        nested_dir = temp_output_dir / "nested" / "output"
        assert not nested_dir.exists()
        
        path = save_forecast_output(
            forecast,
            output_dir=nested_dir,
            year=2025,
            file_format="csv",
        )
        
        assert path.exists()
        assert nested_dir.exists()
    
    def test_save_wrong_year_in_filename(self, config, temp_output_dir):
        """Test that year is reflected in filename."""
        forecast = pd.DataFrame(
            np.random.uniform(1.0, 3.3, (63, 29)),
            index=[f"P{i:02d}" for i in range(1, 64)],
            columns=config.data.all_subcriteria,
        )
        forecast.index.name = "Province"
        
        path = save_forecast_output(
            forecast,
            output_dir=temp_output_dir,
            year=2030,
            file_format="csv",
        )
        
        assert "2030" in path.name


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not HAS_AUTOGLUON, reason="AutoGluon not installed")
class TestIntegration:
    """Integration tests for forecasting pipeline."""
    
    def test_load_to_aggregate_pipeline(
        self,
        synthetic_imputed_panel,
        synthetic_forecasts,
        config,
        temp_output_dir,
    ):
        """Test pipeline: load → build TS → aggregate → validate → save."""
        # Load
        loaded = synthetic_imputed_panel.copy()
        assert loaded.shape == (882, 31)
        
        # Build TimeSeriesDataFrame
        ts_dfs = build_timeseries_dataframes(loaded, config)
        assert len(ts_dfs) == 29
        
        # Aggregate forecasts (using synthetic forecasts)
        agg = aggregate_forecasts(synthetic_forecasts, config)
        assert agg.shape == (63, 29)
        
        # Validate
        result = validate_forecast_output(agg, config)
        assert result is True
        
        # Save
        path = save_forecast_output(agg, output_dir=temp_output_dir)
        assert path.exists()


# =============================================================================
# Edge Cases & Error Handling
# =============================================================================

@pytest.mark.skipif(not HAS_AUTOGLUON, reason="AutoGluon not installed")
class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_forecast_dict(self, config):
        """Test aggregation with empty forecast dictionary."""
        with pytest.raises(Exception):  # IndexError or ForecastingError
            aggregate_forecasts({}, config)
    
    def test_malformed_forecast_province_column(self, config):
        """Test aggregation with malformed Province column."""
        forecasts = {
            "SC11": pd.DataFrame({
                "WrongCol": [f"P{i:02d}" for i in range(1, 64)],
                "forecast": np.random.uniform(1.0, 3.3, 63),
            })
        }
        
        with pytest.raises(Exception):  # KeyError or ForecastingError
            aggregate_forecasts(forecasts, config)
    
    def test_forecast_province_count_mismatch(self, config):
        """Test aggregation when forecasts have different province counts."""
        forecasts = {
            "SC11": pd.DataFrame({
                "Province": [f"P{i:02d}" for i in range(1, 64)],
                "forecast": np.random.uniform(1.0, 3.3, 63),
            }),
            "SC12": pd.DataFrame({
                "Province": [f"P{i:02d}" for i in range(1, 60)],  # Only 59 provinces
                "forecast": np.random.uniform(1.0, 3.3, 59),
            }),
        }
        
        with pytest.raises(Exception):  # ValueError or ForecastingError
            aggregate_forecasts(forecasts, config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
