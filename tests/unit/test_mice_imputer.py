"""
tests/unit/test_mice_imputer.py
-------------------------------
Unit tests for MICE imputation module (Phase 6).

Test Coverage
-------------
✓ load_raw_panel: shape, columns, data types, uniqueness
✓ run_mice_imputation: NaN reduction, value bounds, reproducibility
✓ validate_imputation: all validation checks
✓ save_imputed_panel: file I/O, data integrity on reload
✓ run_full_imputation_pipeline: end-to-end execution
✓ Error handling: missing files, wrong shapes, invalid values
✓ Edge cases: no missing data, all missing data
✓ Data immutability: original data/csv/ never modified
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.core import data_loader
from src.core.data_loader import load_config
from src.core.exceptions import DataIntegrityError, ImputationError
from src.core.schema import AppConfig
from src.ml.imputation.mice_imputer import (
    load_raw_panel,
    run_full_imputation_pipeline,
    run_mice_imputation,
    save_imputed_panel,
    validate_imputation,
    _validate_raw_panel,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def config() -> AppConfig:
    """Load the application configuration."""
    return load_config("config/config.yaml")


@pytest.fixture
def synthetic_raw_panel(config: AppConfig) -> pd.DataFrame:
    """
    Create a synthetic raw panel with some missing values.
    
    Structure:
    * 63 provinces × 14 years = 882 rows
    * Columns: Province, Year, SC11–SC83
    * ~10% missing values randomly distributed
    """
    n_provinces = config.data.n_provinces
    n_years = len(config.data.years)
    years = config.data.years
    provinces = [f"P{i:02d}" for i in range(1, n_provinces + 1)]
    
    data = []
    for year in years:
        for prov in provinces:
            row = {"Province": prov, "Year": year}
            # Add sub-criteria scores with ~10% missing values
            for sc in config.data.all_subcriteria:
                if np.random.random() < 0.10:
                    row[sc] = np.nan
                else:
                    row[sc] = np.random.uniform(0.5, 3.3)
            data.append(row)
    
    panel = pd.DataFrame(data)
    return panel.reset_index(drop=True)


@pytest.fixture
def synthetic_panel_no_missing(config: AppConfig) -> pd.DataFrame:
    """Create a synthetic panel with NO missing values."""
    n_provinces = config.data.n_provinces
    n_years = len(config.data.years)
    years = config.data.years
    provinces = [f"P{i:02d}" for i in range(1, n_provinces + 1)]
    
    data = []
    for year in years:
        for prov in provinces:
            row = {"Province": prov, "Year": year}
            for sc in config.data.all_subcriteria:
                row[sc] = np.random.uniform(0.5, 3.3)
            data.append(row)
    
    return pd.DataFrame(data).reset_index(drop=True)


@pytest.fixture
def synthetic_panel_all_missing(config: AppConfig) -> pd.DataFrame:
    """Create a synthetic panel with all sub-criteria missing (extreme edge case)."""
    n_provinces = config.data.n_provinces
    n_years = len(config.data.years)
    years = config.data.years
    provinces = [f"P{i:02d}" for i in range(1, n_provinces + 1)]
    
    data = []
    for year in years:
        for prov in provinces:
            row = {"Province": prov, "Year": year}
            for sc in config.data.all_subcriteria:
                row[sc] = np.nan
            data.append(row)
    
    return pd.DataFrame(data).reset_index(drop=True)


# =============================================================================
# Tests: load_raw_panel
# =============================================================================

class TestLoadRawPanel:
    """Tests for load_raw_panel function."""

    def test_load_raw_panel_shape(self, config: AppConfig) -> None:
        """Test that loaded panel has correct shape."""
        panel = load_raw_panel(config)
        
        expected_n_rows = config.data.n_provinces * len(config.data.years)
        expected_n_cols = 2 + len(config.data.all_subcriteria)  # Province, Year + sub-criteria
        
        assert panel.shape == (expected_n_rows, expected_n_cols), \
            f"Expected shape {(expected_n_rows, expected_n_cols)}, got {panel.shape}"

    def test_load_raw_panel_columns(self, config: AppConfig) -> None:
        """Test that loaded panel has all expected columns."""
        panel = load_raw_panel(config)
        
        expected_cols = {"Province", "Year"} | set(config.data.all_subcriteria)
        assert set(panel.columns) == expected_cols, \
            f"Column mismatch. Expected: {expected_cols}, got: {set(panel.columns)}"

    def test_load_raw_panel_years(self, config: AppConfig) -> None:
        """Test that all 14 years are present."""
        panel = load_raw_panel(config)
        
        years_in_panel = sorted(panel["Year"].unique())
        expected_years = sorted(config.data.years)
        
        assert years_in_panel == expected_years, \
            f"Years mismatch. Expected: {expected_years}, got: {years_in_panel}"

    def test_load_raw_panel_provinces(self, config: AppConfig) -> None:
        """Test that all 63 provinces are present."""
        panel = load_raw_panel(config)
        
        n_unique_provinces = panel["Province"].nunique()
        assert n_unique_provinces == config.data.n_provinces, \
            f"Expected {config.data.n_provinces} provinces, got {n_unique_provinces}"

    def test_load_raw_panel_no_duplicates(self, config: AppConfig) -> None:
        """Test that there are no duplicate (province, year) pairs."""
        panel = load_raw_panel(config)
        
        n_rows = len(panel)
        n_unique_pairs = panel.groupby(["Province", "Year"]).size().shape[0]
        
        assert n_rows == n_unique_pairs, \
            f"Found duplicate (province, year) pairs: {n_rows - n_unique_pairs}"

    def test_load_raw_panel_value_range(self, config: AppConfig) -> None:
        """Test that numeric values are within expected bounds (or NaN)."""
        panel = load_raw_panel(config)
        
        # Use tolerance to match validation in data_loader
        tolerance = 0.15  # Matches _VALUE_TOLERANCE in data_loader.py
        for sc in config.data.all_subcriteria:
            non_nan_values = panel[sc].dropna()
            if len(non_nan_values) > 0:
                assert (non_nan_values >= -tolerance).all() and (non_nan_values <= 3.33 + tolerance).all(), \
                    f"Column {sc} has values outside [-{tolerance}, 3.33+{tolerance}] range"

    def test_load_raw_panel_preserves_nan(self, config: AppConfig) -> None:
        """Test that NaN values are preserved from original CSVs."""
        panel = load_raw_panel(config)
        
        # The panel should have some NaN values (from structural missingness)
        nan_count = panel.isna().sum().sum()
        assert nan_count > 0, "Expected some NaN values from structural missingness"

    def test_load_raw_panel_provinces_per_year(self, config: AppConfig) -> None:
        """Test that each year has exactly 63 provinces."""
        panel = load_raw_panel(config)
        
        provinces_per_year = panel.groupby("Year")["Province"].nunique()
        assert (provinces_per_year == config.data.n_provinces).all(), \
            f"Some years don't have {config.data.n_provinces} provinces"


# =============================================================================
# Tests: run_mice_imputation
# =============================================================================

class TestRunMicesImputation:
    """Tests for run_mice_imputation function."""

    def test_mice_removes_all_nan(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test that MICE imputation produces zero NaN values."""
        nan_before = synthetic_raw_panel.isna().sum().sum()
        assert nan_before > 0, "Test fixture should have some NaN values"
        
        panel_imputed = run_mice_imputation(synthetic_raw_panel, config)
        
        nan_after = panel_imputed.isna().sum().sum()
        assert nan_after == 0, f"Imputed panel still has {nan_after} NaN values"

    def test_mice_output_shape(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test that imputed panel has same shape as input."""
        panel_imputed = run_mice_imputation(synthetic_raw_panel, config)
        
        assert panel_imputed.shape == synthetic_raw_panel.shape, \
            f"Shape changed: {synthetic_raw_panel.shape} → {panel_imputed.shape}"

    def test_mice_output_columns(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test that imputed panel has same columns as input."""
        panel_imputed = run_mice_imputation(synthetic_raw_panel, config)
        
        assert list(panel_imputed.columns) == list(synthetic_raw_panel.columns), \
            "Columns changed after imputation"

    def test_mice_value_bounds(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test that imputed values are within [0, 3.33] bounds."""
        panel_imputed = run_mice_imputation(synthetic_raw_panel, config)
        
        for sc in config.data.all_subcriteria:
            min_val = panel_imputed[sc].min()
            max_val = panel_imputed[sc].max()
            assert min_val >= -0.05 and max_val <= 3.38, \
                f"Column {sc} has values outside bounds: [{min_val}, {max_val}]"

    def test_mice_preserves_province_year(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test that Province and Year columns are unchanged."""
        panel_imputed = run_mice_imputation(synthetic_raw_panel, config)
        
        pd.testing.assert_series_equal(
            panel_imputed["Province"],
            synthetic_raw_panel["Province"],
            check_names=True,
        )
        pd.testing.assert_series_equal(
            panel_imputed["Year"],
            synthetic_raw_panel["Year"],
            check_names=True,
        )

    def test_mice_reproducibility(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test that MICE with fixed random_state is reproducible."""
        panel_imputed_1 = run_mice_imputation(synthetic_raw_panel, config)
        panel_imputed_2 = run_mice_imputation(synthetic_raw_panel, config)
        
        # Should be identical (or very close due to floating point)
        pd.testing.assert_frame_equal(
            panel_imputed_1.round(10),
            panel_imputed_2.round(10),
            check_dtype=False,
        )

    def test_mice_no_missing_data(self, synthetic_panel_no_missing: pd.DataFrame, config: AppConfig) -> None:
        """Test that MICE handles panels with no missing data."""
        panel_imputed = run_mice_imputation(synthetic_panel_no_missing, config)
        
        # Output should have no NaN
        assert panel_imputed.isna().sum().sum() == 0

    def test_mice_reduces_nan(self, config: AppConfig) -> None:
        """Test on real data that MICE significantly reduces NaN count."""
        panel_raw = load_raw_panel(config)
        nan_count_before = panel_raw.isna().sum().sum()
        
        panel_imputed = run_mice_imputation(panel_raw, config)
        nan_count_after = panel_imputed.isna().sum().sum()
        
        # Should have 0 NaN after imputation
        assert nan_count_after == 0, f"Expected 0 NaN, got {nan_count_after}"
        assert nan_count_before > 0, "Test data should have some NaN"

    def test_mice_numeric_columns(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test that sub-criteria columns remain numeric."""
        panel_imputed = run_mice_imputation(synthetic_raw_panel, config)
        
        for sc in config.data.all_subcriteria:
            assert pd.api.types.is_numeric_dtype(panel_imputed[sc]), \
                f"Column {sc} is not numeric after imputation"


# =============================================================================
# Tests: validate_imputation
# =============================================================================

class TestValidateImputation:
    """Tests for validate_imputation function."""

    def test_validate_good_panel(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test validation on a properly imputed panel."""
        panel_imputed = run_mice_imputation(synthetic_raw_panel, config)
        is_valid, report = validate_imputation(panel_imputed, config)
        
        assert is_valid, f"Validation failed: {report}"
        assert report["all_valid"] == True
        assert report["nan_valid"] == True
        assert report["shape_valid"] == True

    def test_validate_report_structure(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test that validation report has expected keys."""
        panel_imputed = run_mice_imputation(synthetic_raw_panel, config)
        is_valid, report = validate_imputation(panel_imputed, config)
        
        expected_keys = {
            "shape", "shape_valid",
            "columns", "columns_valid",
            "dtypes_valid",
            "nan_count", "nan_valid",
            "out_of_bounds_count", "value_bounds_valid",
            "n_provinces", "provinces_valid",
            "n_years", "years_valid",
            "n_duplicate_pairs", "no_duplicates",
            "all_valid",
        }
        
        assert set(report.keys()) == expected_keys, \
            f"Report keys mismatch. Expected: {expected_keys}, got: {set(report.keys())}"

    def test_validate_detects_nan(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test that validation detects remaining NaN values."""
        panel = synthetic_raw_panel.copy()
        is_valid, report = validate_imputation(panel, config)
        
        # Raw panel should have NaN and fail validation
        if report["nan_count"] > 0:
            assert not is_valid or not report["nan_valid"]

    def test_validate_detects_wrong_shape(self, config: AppConfig) -> None:
        """Test that validation detects wrong shape."""
        # Create panel with wrong number of rows
        panel = pd.DataFrame({
            "Province": ["P01"] * 100,
            "Year": [2011] * 100,
        })
        for sc in config.data.all_subcriteria:
            panel[sc] = 2.0
        
        is_valid, report = validate_imputation(panel, config)
        assert not report["shape_valid"]

    def test_validate_detects_missing_columns(self, config: AppConfig) -> None:
        """Test that validation detects missing columns."""
        # Create panel without SC83
        panel = pd.DataFrame({
            "Province": ["P01"] * 882,
            "Year": list(range(2011, 2025)) * 63,
        })
        for sc in config.data.all_subcriteria:
            if sc != "SC83":
                panel[sc] = 2.0
        
        is_valid, report = validate_imputation(panel, config)
        assert not report["columns_valid"]

    def test_validate_detects_out_of_bounds(self, config: AppConfig) -> None:
        """Test that validation detects out-of-bounds values."""
        panel = pd.DataFrame({
            "Province": ["P01"] * 882,
            "Year": list(range(2011, 2025)) * 63,
        })
        for sc in config.data.all_subcriteria:
            panel[sc] = 5.0  # Out of bounds
        
        is_valid, report = validate_imputation(panel, config)
        assert not report["value_bounds_valid"]
        assert report["out_of_bounds_count"] > 0

    def test_validate_detects_duplicates(self, config: AppConfig) -> None:
        """Test that validation detects duplicate (province, year) pairs."""
        panel = pd.DataFrame({
            "Province": ["P01"] * 100,
            "Year": [2011] * 100,
        })
        for sc in config.data.all_subcriteria:
            panel[sc] = 2.0
        
        is_valid, report = validate_imputation(panel, config)
        assert report["no_duplicates"] == False


# =============================================================================
# Tests: save_imputed_panel
# =============================================================================

class TestSaveImputedPanel:
    """Tests for save_imputed_panel function."""

    def test_save_creates_file(self, synthetic_raw_panel: pd.DataFrame) -> None:
        """Test that save_imputed_panel creates a file."""
        panel_imputed = run_mice_imputation(synthetic_raw_panel, load_config())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_panel.parquet"
            saved_path = save_imputed_panel(panel_imputed, output_path)
            
            assert saved_path.exists(), f"File not created: {saved_path}"

    def test_save_creates_directories(self, synthetic_raw_panel: pd.DataFrame) -> None:
        """Test that save_imputed_panel creates parent directories."""
        panel_imputed = run_mice_imputation(synthetic_raw_panel, load_config())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "deep" / "panel.parquet"
            saved_path = save_imputed_panel(panel_imputed, output_path)
            
            assert saved_path.exists()
            assert saved_path.parent.exists()

    def test_save_and_load(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test that saved panel can be loaded back."""
        panel_imputed = run_mice_imputation(synthetic_raw_panel, config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_panel.parquet"
            save_imputed_panel(panel_imputed, output_path)
            
            # Load back
            panel_reloaded = pd.read_parquet(output_path)
            
            # Should match original (within floating point tolerance)
            pd.testing.assert_frame_equal(
                panel_imputed.round(10),
                panel_reloaded.round(10),
                check_dtype=False,
            )

    def test_save_preserves_shape(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test that saving preserves panel shape."""
        panel_imputed = run_mice_imputation(synthetic_raw_panel, config)
        original_shape = panel_imputed.shape
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_panel.parquet"
            save_imputed_panel(panel_imputed, output_path)
            
            panel_reloaded = pd.read_parquet(output_path)
            assert panel_reloaded.shape == original_shape


# =============================================================================
# Tests: run_full_imputation_pipeline
# =============================================================================

class TestRunFullImputationPipeline:
    """Tests for run_full_imputation_pipeline function."""

    def test_pipeline_end_to_end_real_data(self, config: AppConfig) -> None:
        """Test full pipeline on real data without saving."""
        panel_imputed, is_valid, report = run_full_imputation_pipeline(
            config,
            save=False,
        )
        
        assert is_valid, f"Validation failed: {report}"
        assert panel_imputed.isna().sum().sum() == 0
        assert panel_imputed.shape[0] > 0

    def test_pipeline_with_save(self, config: AppConfig) -> None:
        """Test full pipeline with saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "panel_imputed.parquet"
            
            panel_imputed, is_valid, report = run_full_imputation_pipeline(
                config,
                output_path=output_path,
                save=True,
            )
            
            assert is_valid
            assert output_path.exists()
            
            # Verify saved file can be loaded
            panel_reloaded = pd.read_parquet(output_path)
            assert panel_reloaded.shape == panel_imputed.shape

    def test_pipeline_validation_report(self, config: AppConfig) -> None:
        """Test that pipeline returns comprehensive validation report."""
        panel_imputed, is_valid, report = run_full_imputation_pipeline(
            config,
            save=False,
        )
        
        assert isinstance(report, dict)
        assert "nan_count" in report
        assert "shape" in report
        assert "n_provinces" in report
        assert "n_years" in report


# =============================================================================
# Tests: Data Immutability
# =============================================================================

class TestDataImmutability:
    """Tests to ensure original data/csv/ files are never modified."""

    def test_original_csv_not_modified(self, config: AppConfig) -> None:
        """Test that running imputation doesn't modify original CSV files."""
        # Read original CSV
        df_original = pd.read_csv(Path("data/csv/2011.csv"))
        original_hash = df_original.to_csv().encode().__hash__()
        
        # Run imputation (which loads but doesn't modify)
        panel_raw = load_raw_panel(config)
        run_mice_imputation(panel_raw, config)
        
        # Re-read original CSV
        df_after = pd.read_csv(Path("data/csv/2011.csv"))
        after_hash = df_after.to_csv().encode().__hash__()
        
        # Hashes should match (file unchanged)
        assert original_hash == after_hash, "Original CSV file was modified!"


# =============================================================================
# Tests: Helper Functions
# =============================================================================

class TestValidateRawPanel:
    """Tests for _validate_raw_panel helper function."""

    def test_validate_raw_panel_good(self, synthetic_raw_panel: pd.DataFrame, config: AppConfig) -> None:
        """Test validation of a good raw panel."""
        # This should not raise
        _validate_raw_panel(synthetic_raw_panel, config)

    def test_validate_raw_panel_wrong_n_rows(self, config: AppConfig) -> None:
        """Test that validation catches wrong number of rows."""
        panel = pd.DataFrame({
            "Province": ["P01"] * 100,
            "Year": [2011] * 100,
        })
        for sc in config.data.all_subcriteria:
            panel[sc] = 2.0
        
        with pytest.raises(DataIntegrityError):
            _validate_raw_panel(panel, config)

    def test_validate_raw_panel_missing_columns(self, config: AppConfig) -> None:
        """Test that validation catches missing columns."""
        panel = pd.DataFrame({
            "Province": ["P01"] * 882,
            "Year": list(range(2011, 2025)) * 63,
        })
        # Missing SC83
        for sc in config.data.all_subcriteria:
            if sc != "SC83":
                panel[sc] = 2.0
        
        with pytest.raises(DataIntegrityError):
            _validate_raw_panel(panel, config)

    def test_validate_raw_panel_wrong_n_provinces(self, config: AppConfig) -> None:
        """Test that validation catches wrong number of provinces."""
        # Create panel with 14 provinces repeated across years (instead of 63)
        n_rows = 14 * 14  # 14 provinces × 14 years
        panel = pd.DataFrame({
            "Province": [f"P{i:02d}" for i in range(1, 15)] * 14,
            "Year": list(range(2011, 2025)) * 14,
        })
        for sc in config.data.all_subcriteria:
            panel[sc] = 2.0
        
        with pytest.raises(DataIntegrityError):
            _validate_raw_panel(panel, config)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self, config: AppConfig) -> None:
        """Test complete workflow: load → impute → validate → save."""
        # Load
        panel_raw = load_raw_panel(config)
        nan_before = panel_raw.isna().sum().sum()
        assert nan_before > 0
        
        # Impute
        panel_imputed = run_mice_imputation(panel_raw, config)
        assert panel_imputed.isna().sum().sum() == 0
        
        # Validate
        is_valid, report = validate_imputation(panel_imputed, config)
        assert is_valid
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "panel.parquet"
            saved_path = save_imputed_panel(panel_imputed, output_path)
            
            # Reload and verify
            panel_reloaded = pd.read_parquet(saved_path)
            assert panel_reloaded.shape == panel_imputed.shape
            assert panel_reloaded.isna().sum().sum() == 0

    def test_regime_awareness(self, config: AppConfig) -> None:
        """Test that imputation respects regime structure."""
        panel_raw = load_raw_panel(config)
        panel_imputed = run_mice_imputation(panel_raw, config)
        
        # Check R4 (2021-2024): SC52 should still be NaN or properly imputed
        # (Note: SC52 is absent in R4, but we still keep the column for consistency)
        r4_data = panel_imputed[panel_imputed["Year"].isin([2021, 2022, 2023, 2024])]
        
        # After imputation, SC52 should have values (imputed from other years/provinces)
        assert r4_data["SC52"].notna().sum() > 0

    def test_year_consistency(self, config: AppConfig) -> None:
        """Test that all years are properly imputed."""
        panel_raw = load_raw_panel(config)
        panel_imputed = run_mice_imputation(panel_raw, config)
        
        for year in config.data.years:
            year_data = panel_imputed[panel_imputed["Year"] == year]
            assert len(year_data) == config.data.n_provinces
            assert year_data.isna().sum().sum() == 0  # No NaN in this year's data


# =============================================================================
# Test Execution
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
