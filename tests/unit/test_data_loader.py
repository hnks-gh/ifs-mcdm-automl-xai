"""
tests/unit/test_data_loader.py
------------------------------
Unit tests for src/core/data_loader.py.

Tests are designed to be runnable WITHOUT the real data files by using
synthetic fixtures.  Tests that exercise the real CSV files are marked
``@pytest.mark.integration`` and skipped in standard CI.
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.data_loader import (
    _validate_year_dataframe,
    compute_missingness_report,
    detect_regimes,
    get_active_subcriteria_for_year,
    get_regime_for_year,
    load_codebook,
    load_config,
    load_year,
)
from src.core.exceptions import (
    ConfigurationError,
    DataIntegrityError,
    DataLoadError,
    RegimeDetectionError,
)
from src.core.schema import AppConfig, Regime, RegimeConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def minimal_df() -> pd.DataFrame:
    """A minimal valid province-indexed DataFrame (3 provinces, 3 sub-criteria)."""
    data = {
        "SC11": [1.0, 2.0, 3.0],
        "SC12": [0.5, 1.5, 2.5],
        "SC13": [0.0, np.nan, 1.0],
    }
    df = pd.DataFrame(data, index=["P01", "P02", "P03"])
    df.index.name = "Province"
    return df


@pytest.fixture()
def full_regime_config() -> Dict[str, RegimeConfig]:
    """Minimal regime config for 2 years with 2 different active sub-criteria sets."""
    return {
        "R1": RegimeConfig(
            years=[2019, 2020],
            absent_subcriteria=[],
            active_subcriteria=["SC11", "SC12"],
            n_active=2,
        ),
        "R2": RegimeConfig(
            years=[2021],
            absent_subcriteria=["SC12"],
            active_subcriteria=["SC11"],
            n_active=1,
        ),
    }


@pytest.fixture()
def minimal_panel(minimal_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Panel with 2 years using the same DataFrame."""
    df1 = minimal_df.copy()
    df2 = minimal_df[["SC11", "SC12"]].copy()  # SC13 absent
    return {2019: df1, 2020: df2}


# =============================================================================
# _validate_year_dataframe
# =============================================================================

class TestValidateYearDataFrame:

    def test_valid_dataframe_passes(self, minimal_df: pd.DataFrame) -> None:
        """A well-formed DataFrame should not raise."""
        _validate_year_dataframe(minimal_df, year=2019)

    def test_too_many_rows_raises(self) -> None:
        """More than 63 province rows should raise DataIntegrityError."""
        data = {f"SC11": [1.0] * 64}
        df = pd.DataFrame(data)
        df.index.name = "Province"
        with pytest.raises(DataIntegrityError, match="province rows"):
            _validate_year_dataframe(df, year=2019)

    def test_unknown_column_raises(self) -> None:
        """Columns not in the known 29 sub-criteria set should raise."""
        df = pd.DataFrame(
            {"SC11": [1.0], "UNKNOWN_COL": [2.0]},
            index=["P01"],
        )
        df.index.name = "Province"
        with pytest.raises(DataIntegrityError, match="unrecognised columns"):
            _validate_year_dataframe(df, year=2019)

    def test_value_above_max_raises(self) -> None:
        """Values > 3.38 (3.33 + tolerance) should raise DataIntegrityError."""
        df = pd.DataFrame({"SC11": [5.0]}, index=["P01"])
        df.index.name = "Province"
        with pytest.raises(DataIntegrityError, match="outside"):
            _validate_year_dataframe(df, year=2019)

    def test_value_below_min_raises(self) -> None:
        """Negative values should raise DataIntegrityError."""
        df = pd.DataFrame({"SC11": [-0.5]}, index=["P01"])
        df.index.name = "Province"
        with pytest.raises(DataIntegrityError, match="outside"):
            _validate_year_dataframe(df, year=2019)

    def test_nan_values_are_allowed(self) -> None:
        """NaN values (missing data) should NOT raise."""
        df = pd.DataFrame({"SC11": [np.nan, 1.0]}, index=["P01", "P02"])
        df.index.name = "Province"
        _validate_year_dataframe(df, year=2019)  # must not raise

    def test_all_nan_column_is_allowed(self) -> None:
        """An all-NaN column (structural absence) should NOT raise."""
        df = pd.DataFrame({"SC11": [1.0], "SC12": [np.nan]}, index=["P01"])
        df.index.name = "Province"
        _validate_year_dataframe(df, year=2019)  # must not raise

    def test_boundary_value_at_max_passes(self) -> None:
        """Exactly 3.33 is within bounds."""
        df = pd.DataFrame({"SC11": [3.33]}, index=["P01"])
        df.index.name = "Province"
        _validate_year_dataframe(df, year=2019)

    def test_boundary_value_zero_passes(self) -> None:
        """Exactly 0.0 is within bounds."""
        df = pd.DataFrame({"SC11": [0.0]}, index=["P01"])
        df.index.name = "Province"
        _validate_year_dataframe(df, year=2019)


# =============================================================================
# detect_regimes
# =============================================================================

class TestDetectRegimes:

    def test_valid_regime_config_passes(self) -> None:
        """Detected active columns match config → no error, correct Regime objects."""
        panel = {
            2019: pd.DataFrame({"SC11": [1.0], "SC12": [2.0]}, index=["P01"]),
            2021: pd.DataFrame({"SC11": [1.0]}, index=["P01"]),
        }
        config_regimes = {
            "R1": RegimeConfig(
                years=[2019], absent_subcriteria=["SC12"],
                active_subcriteria=["SC11", "SC12"], n_active=2,
            ),
        }
        # Panel year 2021 not in R1 — only validate years in config
        # Use only year 2019 in config
        panel_sub = {2019: panel[2019]}
        config_regimes_sub = {
            "R1": RegimeConfig(
                years=[2019], absent_subcriteria=[],
                active_subcriteria=["SC11", "SC12"], n_active=2,
            ),
        }
        regimes = detect_regimes(panel_sub, config_regimes=config_regimes_sub)
        assert "R1" in regimes
        assert regimes["R1"].n_active == 2
        assert 2019 in regimes["R1"].years

    def test_regime_mismatch_raises(self) -> None:
        """Active columns not matching config should raise RegimeDetectionError."""
        # Config says SC12 should be active, but panel has SC12 all-NaN
        panel = {
            2019: pd.DataFrame(
                {"SC11": [1.0], "SC12": [np.nan]}, index=["P01"]
            ),
        }
        config_regimes = {
            "R1": RegimeConfig(
                years=[2019], absent_subcriteria=[],
                active_subcriteria=["SC11", "SC12"], n_active=2,
            ),
        }
        with pytest.raises(RegimeDetectionError):
            detect_regimes(panel, config_regimes=config_regimes)

    def test_auto_infer_regimes(self) -> None:
        """Without config, regimes are auto-detected by column fingerprint."""
        panel = {
            2019: pd.DataFrame({"SC11": [1.0], "SC12": [2.0]}, index=["P01"]),
            2020: pd.DataFrame({"SC11": [1.0], "SC12": [2.0]}, index=["P01"]),
            2021: pd.DataFrame({"SC11": [1.0], "SC12": [np.nan]}, index=["P01"]),
        }
        regimes = detect_regimes(panel, config_regimes=None)
        # Two distinct fingerprints → 2 regimes
        assert len(regimes) == 2
        # 2019 and 2020 share the same fingerprint → same regime
        all_years = {yr for r in regimes.values() for yr in r.years}
        assert 2019 in all_years
        assert 2020 in all_years
        assert 2021 in all_years

    def test_single_year_single_regime(self) -> None:
        """Single year should produce a single regime."""
        panel = {
            2019: pd.DataFrame({"SC11": [1.0]}, index=["P01"]),
        }
        regimes = detect_regimes(panel, config_regimes=None)
        assert len(regimes) == 1


# =============================================================================
# get_regime_for_year
# =============================================================================

class TestGetRegimeForYear:

    @pytest.fixture()
    def sample_regimes(self) -> Dict[str, Regime]:
        return {
            "R1": Regime(
                regime_id="R1",
                years=[2011, 2012, 2013],
                active_subcriteria=["SC11", "SC12"],
                absent_subcriteria=["SC13"],
            ),
            "R2": Regime(
                regime_id="R2",
                years=[2014, 2015],
                active_subcriteria=["SC11", "SC12", "SC13"],
                absent_subcriteria=[],
            ),
        }

    def test_returns_correct_regime(self, sample_regimes: Dict[str, Regime]) -> None:
        regime = get_regime_for_year(2012, sample_regimes)
        assert regime.regime_id == "R1"

    def test_returns_correct_regime_r2(self, sample_regimes: Dict[str, Regime]) -> None:
        regime = get_regime_for_year(2015, sample_regimes)
        assert regime.regime_id == "R2"

    def test_unknown_year_raises(self, sample_regimes: Dict[str, Regime]) -> None:
        with pytest.raises(RegimeDetectionError, match="No regime found"):
            get_regime_for_year(2099, sample_regimes)


# =============================================================================
# get_active_subcriteria_for_year
# =============================================================================

class TestGetActiveSubcriteriaForYear:

    @pytest.fixture()
    def regimes(self) -> Dict[str, Regime]:
        return {
            "R1": Regime(
                regime_id="R1",
                years=[2011],
                active_subcriteria=["SC11", "SC12"],
                absent_subcriteria=["SC13"],
            ),
        }

    def test_returns_active_subcriteria(self, regimes: Dict[str, Regime]) -> None:
        result = get_active_subcriteria_for_year(2011, regimes)
        assert result == ["SC11", "SC12"]

    def test_unknown_year_raises(self, regimes: Dict[str, Regime]) -> None:
        with pytest.raises(RegimeDetectionError):
            get_active_subcriteria_for_year(2099, regimes)


# =============================================================================
# compute_missingness_report
# =============================================================================

class TestComputeMissingnessReport:

    def test_no_missing_data(self) -> None:
        panel = {
            2019: pd.DataFrame({"SC11": [1.0, 2.0]}, index=["P01", "P02"]),
        }
        report = compute_missingness_report(panel)
        assert len(report) == 1
        row = report.iloc[0]
        assert row["n_missing"] == 0
        assert row["pct_missing"] == 0.0

    def test_partial_missing(self) -> None:
        panel = {
            2019: pd.DataFrame(
                {"SC11": [1.0, np.nan], "SC12": [np.nan, np.nan]},
                index=["P01", "P02"],
            ),
        }
        report = compute_missingness_report(panel)
        sc11_row = report[report["subcriteria"] == "SC11"].iloc[0]
        sc12_row = report[report["subcriteria"] == "SC12"].iloc[0]
        assert sc11_row["n_missing"] == 1
        assert sc11_row["pct_missing"] == 50.0
        assert sc12_row["n_missing"] == 2
        assert sc12_row["pct_missing"] == 100.0

    def test_multiple_years(self) -> None:
        panel = {
            2019: pd.DataFrame({"SC11": [1.0]}, index=["P01"]),
            2020: pd.DataFrame({"SC11": [np.nan]}, index=["P01"]),
        }
        report = compute_missingness_report(panel)
        assert len(report) == 2
        year_2019 = report[report["year"] == 2019].iloc[0]
        year_2020 = report[report["year"] == 2020].iloc[0]
        assert year_2019["n_missing"] == 0
        assert year_2020["n_missing"] == 1

    def test_output_columns(self) -> None:
        panel = {2019: pd.DataFrame({"SC11": [1.0]}, index=["P01"])}
        report = compute_missingness_report(panel)
        assert set(report.columns) == {"year", "subcriteria", "n_missing", "pct_missing"}


# =============================================================================
# load_config — error paths (file system mocked)
# =============================================================================

class TestLoadConfig:

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(config_path=tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "config.yaml"
        bad_yaml.write_text("data: {unclosed_brace", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="parse"):
            load_config(config_path=bad_yaml)


# =============================================================================
# load_year — error paths (file system mocked)
# =============================================================================

class TestLoadYear:

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataLoadError, match="not found"):
            load_year(2019, csv_dir=tmp_path)

    def test_missing_province_column_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "2019.csv"
        csv_path.write_text("SC11,SC12\n1.0,2.0\n", encoding="utf-8")
        with pytest.raises(DataIntegrityError, match="Province column"):
            load_year(2019, csv_dir=tmp_path)

    def test_valid_csv_loads_correctly(self, tmp_path: Path) -> None:
        csv_content = "Province,SC11,SC12\nP01,1.0,2.0\nP02,0.5,1.5\n"
        csv_path = tmp_path / "2019.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        df = load_year(2019, csv_dir=tmp_path)
        assert df.index.name == "Province"
        assert list(df.index) == ["P01", "P02"]
        assert "SC11" in df.columns
        assert "SC12" in df.columns
        assert df.loc["P01", "SC11"] == pytest.approx(1.0)

    def test_non_numeric_values_become_nan(self, tmp_path: Path) -> None:
        csv_content = "Province,SC11\nP01,N/A\nP02,1.0\n"
        csv_path = tmp_path / "2019.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        df = load_year(2019, csv_dir=tmp_path)
        assert pd.isna(df.loc["P01", "SC11"])
        assert df.loc["P02", "SC11"] == pytest.approx(1.0)


# =============================================================================
# load_codebook — error paths
# =============================================================================

class TestLoadCodebook:

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataLoadError, match="not found"):
            load_codebook(codebook_dir=tmp_path)

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        # Create codebook files, but provinces file is missing a required column
        (tmp_path / "codebook_provinces.csv").write_text(
            "Variable_Code\nP01\n", encoding="utf-8"
        )
        (tmp_path / "codebook_criteria.csv").write_text(
            "Variable_Code,Variable_Name\nC01,Participation\n", encoding="utf-8"
        )
        (tmp_path / "codebook_subcriteria.csv").write_text(
            "Variable_Code,Variable_Name,Criteria_Name,Criteria_Code\n"
            "SC11,Civic Knowledge,Participation,C01\n",
            encoding="utf-8",
        )
        with pytest.raises(DataIntegrityError, match="missing columns"):
            load_codebook(codebook_dir=tmp_path)

    def test_valid_codebook_loads(self, tmp_path: Path) -> None:
        (tmp_path / "codebook_provinces.csv").write_text(
            "Variable_Code,Variable_Name\nP01,Hanoi\n", encoding="utf-8"
        )
        (tmp_path / "codebook_criteria.csv").write_text(
            "Variable_Code,Variable_Name\nC01,Participation\n", encoding="utf-8"
        )
        (tmp_path / "codebook_subcriteria.csv").write_text(
            "Variable_Code,Variable_Name,Criteria_Name,Criteria_Code\n"
            "SC11,Civic Knowledge,Participation,C01\n",
            encoding="utf-8",
        )
        codebook = load_codebook(codebook_dir=tmp_path)
        assert "provinces" in codebook
        assert "criteria" in codebook
        assert "subcriteria" in codebook
        assert codebook["provinces"].loc[0, "Variable_Name"] == "Hanoi"
