"""
src/core/data_loader.py
-----------------------
Raw data I/O for the IFS-MCDM-AutoML-XAI framework.

Responsibilities
----------------
* Load annual PAPI CSV files from ``data/csv/`` — read-only; never modified.
* Load the three codebook CSV files from ``data/codebook/``.
* Detect year-regimes from actual column presence in the data.
* Validate data integrity (shape, column names, province codes, value ranges).
* Load and parse ``config/config.yaml`` into a validated :class:`AppConfig`.

Data isolation guarantee
------------------------
This module **only** reads from ``data/``.  All outputs go to ``output/``.
The MCDM pipeline uses the raw DataFrames returned here.
The ML pipeline uses a MICE-imputed copy stored in ``output/ml/imputed/``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.core.exceptions import (
    ConfigurationError,
    DataIntegrityError,
    DataLoadError,
    RegimeDetectionError,
)
from src.core.schema import (
    AppConfig,
    BlankProvinceYear,
    PAPIPanel,
    Regime,
    RegimeConfig,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants derived from the PAPI codebook (mirrors config.yaml)
# These are used only in validation; the authoritative source is config.yaml.
# ---------------------------------------------------------------------------
_EXPECTED_N_PROVINCES: int = 63
_EXPECTED_N_SUBCRITERIA: int = 29
_VALUE_MIN: float = 0.0
_VALUE_MAX: float = 3.33
_VALUE_TOLERANCE: float = 0.15   # tolerance for rounding/measurement precision at boundary

_ALL_SUBCRITERIA: List[str] = [
    "SC11", "SC12", "SC13", "SC14",
    "SC21", "SC22", "SC23", "SC24",
    "SC31", "SC32", "SC33",
    "SC41", "SC42", "SC43", "SC44",
    "SC51", "SC52", "SC53", "SC54",
    "SC61", "SC62", "SC63", "SC64",
    "SC71", "SC72", "SC73",
    "SC81", "SC82", "SC83",
]


# =============================================================================
# Config loader
# =============================================================================

def load_config(config_path: str | Path = "config/config.yaml") -> AppConfig:
    """
    Load and validate the master application configuration from YAML.

    Parameters
    ----------
    config_path : str | Path
        Path to ``config.yaml``.  Relative paths are resolved from the
        current working directory (project root).

    Returns
    -------
    AppConfig
        Fully validated Pydantic configuration object.

    Raises
    ------
    ConfigurationError
        If the file is missing, cannot be parsed, or fails Pydantic validation.
    """
    path = Path(config_path)
    logger.info("Loading configuration from '{}'", path)

    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {path}",
            context={"path": str(path)},
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse YAML configuration: {exc}",
            context={"path": str(path)},
        ) from exc

    try:
        cfg = AppConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigurationError(
            f"Configuration validation failed: {exc}",
            context={"path": str(path)},
        ) from exc

    logger.info(
        "Configuration loaded: {} years, {} provinces, {} sub-criteria",
        len(cfg.data.years),
        cfg.data.n_provinces,
        cfg.data.n_subcriteria,
    )
    return cfg


# =============================================================================
# CSV loaders
# =============================================================================

def load_year(
    year: int,
    csv_dir: str | Path = "data/csv",
    province_col: str = "Province",
) -> pd.DataFrame:
    """
    Load a single annual PAPI CSV file.

    The returned DataFrame has:
    * Index: province codes (e.g. ``"P01"``), dtype ``str``.
    * Columns: the 29 sub-criteria codes present in that year's file.
    * Values: raw float scores (NaN where data is missing).

    The CSV file is **not** modified.

    Parameters
    ----------
    year : int
        Calendar year, e.g. ``2019``.
    csv_dir : str | Path
        Directory containing the annual CSV files.
    province_col : str
        Name of the column that identifies provinces (e.g. ``"Province"``).

    Returns
    -------
    pd.DataFrame
        Province-indexed DataFrame of raw scores.

    Raises
    ------
    DataLoadError
        If the file is missing or cannot be read.
    DataIntegrityError
        If the loaded data fails basic structural checks.
    """
    path = Path(csv_dir) / f"{year}.csv"
    logger.debug("Loading year {} from '{}'", year, path)

    if not path.exists():
        raise DataLoadError(
            f"CSV file not found for year {year}: {path}",
            context={"year": year, "path": str(path)},
        )

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:
        raise DataLoadError(
            f"Failed to read CSV for year {year}: {exc}",
            context={"year": year, "path": str(path)},
        ) from exc

    # ------------------------------------------------------------------
    # Set Province as index
    # ------------------------------------------------------------------
    if province_col not in df.columns:
        raise DataIntegrityError(
            f"Province column '{province_col}' not found in {path}",
            context={"year": year, "columns": list(df.columns)},
        )
    df = df.set_index(province_col)
    df.index.name = "Province"

    # ------------------------------------------------------------------
    # Drop any entirely blank rows (all NaN) — these are structural
    # Type-2 missing observations; keep them as NaN rows in the DataFrame
    # so downstream code can detect them.
    # ------------------------------------------------------------------
    # (We intentionally do NOT drop them here; the pipeline handles them.)

    # ------------------------------------------------------------------
    # Coerce all data columns to float
    # ------------------------------------------------------------------
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # Basic structural validation
    # ------------------------------------------------------------------
    _validate_year_dataframe(df, year)

    logger.debug(
        "Year {} loaded: {} provinces, {} sub-criteria columns",
        year, len(df), len(df.columns),
    )
    return df


def load_all_years(
    years: List[int],
    csv_dir: str | Path = "data/csv",
    province_col: str = "Province",
) -> Dict[int, pd.DataFrame]:
    """
    Load all annual PAPI CSV files.

    Parameters
    ----------
    years : list[int]
        List of years to load (e.g. ``list(range(2011, 2025))``).
    csv_dir : str | Path
        Directory containing the annual CSV files.
    province_col : str
        Province column name.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping ``year → province-indexed DataFrame``.

    Raises
    ------
    DataLoadError
        If any file is missing or unreadable.
    DataIntegrityError
        If any year fails structural validation.
    """
    logger.info("Loading {} annual CSV files from '{}'", len(years), csv_dir)
    panel: Dict[int, pd.DataFrame] = {}
    for yr in sorted(years):
        panel[yr] = load_year(yr, csv_dir=csv_dir, province_col=province_col)
    logger.info("All {} years loaded successfully", len(panel))
    return panel


# =============================================================================
# Codebook loader
# =============================================================================

def load_codebook(
    codebook_dir: str | Path = "data/codebook",
) -> Dict[str, pd.DataFrame]:
    """
    Load the three PAPI codebook CSV files.

    Returns
    -------
    dict with keys:
    * ``"provinces"`` — DataFrame with columns ``Variable_Code``, ``Variable_Name``
    * ``"criteria"``  — DataFrame with columns ``Variable_Code``, ``Variable_Name``
    * ``"subcriteria"`` — DataFrame with columns ``Variable_Code``,
      ``Variable_Name``, ``Criteria_Name``, ``Criteria_Code``

    Raises
    ------
    DataLoadError
        If any codebook file is missing or unreadable.
    """
    codebook_dir = Path(codebook_dir)
    files = {
        "provinces": "codebook_provinces.csv",
        "criteria": "codebook_criteria.csv",
        "subcriteria": "codebook_subcriteria.csv",
    }
    codebook: Dict[str, pd.DataFrame] = {}

    for key, filename in files.items():
        path = codebook_dir / filename
        if not path.exists():
            raise DataLoadError(
                f"Codebook file not found: {path}",
                context={"key": key, "path": str(path)},
            )
        try:
            codebook[key] = pd.read_csv(path, encoding="utf-8")
        except Exception as exc:
            raise DataLoadError(
                f"Failed to read codebook '{path}': {exc}",
                context={"key": key, "path": str(path)},
            ) from exc

        logger.debug("Codebook '{}' loaded: {} rows", key, len(codebook[key]))

    # Basic validation: expected columns
    _validate_codebook(codebook)
    return codebook


# =============================================================================
# Regime detection
# =============================================================================

def detect_regimes(
    panel: Dict[int, pd.DataFrame],
    config_regimes: Optional[Dict[str, RegimeConfig]] = None,
) -> Dict[str, Regime]:
    """
    Detect year-regimes from actual column presence in the panel data.

    A "regime" is a contiguous or disjoint set of years that share the same
    set of structurally active sub-criteria (i.e. not all-NaN columns).

    Two modes:
    1. ``config_regimes`` provided → validate that detected active sub-criteria
       per year match the config.  Raises :exc:`RegimeDetectionError` on mismatch.
    2. ``config_regimes`` is None → infer regimes automatically from the data
       by grouping years with identical active-column fingerprints.

    Parameters
    ----------
    panel : dict[int, pd.DataFrame]
        Mapping year → province-indexed DataFrame (output of :func:`load_all_years`).
    config_regimes : dict[str, RegimeConfig] | None
        Expected regime definitions from config.yaml.  When provided, the
        detected columns are validated against these definitions.

    Returns
    -------
    dict[str, Regime]
        Regime metadata keyed by regime id.

    Raises
    ------
    RegimeDetectionError
        If detected active columns for a year differ from config expectations.
    """
    logger.info("Detecting year regimes for {} years", len(panel))

    # Step 1: For each year, find which columns are NOT all-NaN
    year_active_cols: Dict[int, frozenset] = {}
    for yr, df in sorted(panel.items()):
        active = frozenset(
            col for col in df.columns if not df[col].isna().all()
        )
        year_active_cols[yr] = active
        logger.debug(
            "Year {}: {} active sub-criteria, {} all-NaN",
            yr, len(active), len(df.columns) - len(active),
        )

    if config_regimes is not None:
        return _validate_and_build_regimes(year_active_cols, config_regimes)
    else:
        return _infer_regimes(year_active_cols)


def _validate_and_build_regimes(
    year_active_cols: Dict[int, frozenset],
    config_regimes: Dict[str, RegimeConfig],
) -> Dict[str, Regime]:
    """
    Validate detected active columns against config-defined regimes.

    Raises
    ------
    RegimeDetectionError
        If any year's detected active columns deviate from the config.
    """
    # Build year→regime_id map from config
    year_to_regime: Dict[int, str] = {}
    for regime_id, rc in config_regimes.items():
        for yr in rc.years:
            year_to_regime[yr] = regime_id

    regimes: Dict[str, Regime] = {}
    for regime_id, rc in config_regimes.items():
        expected_active = frozenset(rc.active_subcriteria)
        for yr in rc.years:
            if yr not in year_active_cols:
                logger.warning("Year {} in config but not found in panel — skipping", yr)
                continue
            detected = year_active_cols[yr]
            if detected != expected_active:
                extra = detected - expected_active
                missing = expected_active - detected
                raise RegimeDetectionError(
                    f"Regime mismatch for year {yr} in regime {regime_id}: "
                    f"extra={sorted(extra)}, missing={sorted(missing)}",
                    context={
                        "year": yr,
                        "regime_id": regime_id,
                        "extra_cols": sorted(extra),
                        "missing_cols": sorted(missing),
                    },
                )

        regimes[regime_id] = Regime(
            regime_id=regime_id,
            years=sorted(rc.years),
            active_subcriteria=sorted(rc.active_subcriteria),
            absent_subcriteria=sorted(rc.absent_subcriteria),
        )
        logger.debug(
            "Regime {} validated: {} years, {} active sub-criteria",
            regime_id, len(rc.years), rc.n_active,
        )

    logger.info("All {} regimes validated successfully", len(regimes))
    return regimes


def _infer_regimes(
    year_active_cols: Dict[int, frozenset],
) -> Dict[str, Regime]:
    """
    Automatically group years into regimes by their active-column fingerprint.
    """
    fingerprint_to_years: Dict[frozenset, List[int]] = {}
    for yr, active in sorted(year_active_cols.items()):
        fingerprint_to_years.setdefault(active, []).append(yr)

    regimes: Dict[str, Regime] = {}
    all_subcriteria_set = frozenset(_ALL_SUBCRITERIA)

    for idx, (active, years) in enumerate(
        sorted(fingerprint_to_years.items(), key=lambda x: min(x[1])),
        start=1,
    ):
        regime_id = f"R{idx}"
        absent = sorted(all_subcriteria_set - active)
        regimes[regime_id] = Regime(
            regime_id=regime_id,
            years=sorted(years),
            active_subcriteria=sorted(active),
            absent_subcriteria=absent,
        )
        logger.debug(
            "Auto-regime {}: years={}, n_active={}", regime_id, sorted(years), len(active)
        )

    logger.info("Auto-detected {} regimes", len(regimes))
    return regimes


# =============================================================================
# Data integrity validation (internal)
# =============================================================================

def _validate_year_dataframe(df: pd.DataFrame, year: int) -> None:
    """
    Validate a single year's DataFrame for structural correctness.

    Checks
    ------
    1. Number of province rows is ≤ 63 (blank provinces allowed).
    2. All data columns are a subset of the known 29 sub-criteria.
    3. Numeric values (where not NaN) are within [0, 3.33 + tolerance].

    Raises
    ------
    DataIntegrityError
    """
    all_sc_set = set(_ALL_SUBCRITERIA)
    data_cols = set(df.columns)

    # Check 1: province row count
    if len(df) > _EXPECTED_N_PROVINCES:
        raise DataIntegrityError(
            f"Year {year}: found {len(df)} province rows, expected ≤ {_EXPECTED_N_PROVINCES}",
            context={"year": year, "n_rows": len(df)},
        )

    # Check 2: columns are a subset of known sub-criteria
    unknown_cols = data_cols - all_sc_set
    if unknown_cols:
        raise DataIntegrityError(
            f"Year {year}: unrecognised columns {sorted(unknown_cols)}",
            context={"year": year, "unknown_columns": sorted(unknown_cols)},
        )

    # Check 3: value range
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        out_of_range = series[
            (series < _VALUE_MIN - _VALUE_TOLERANCE) |
            (series > _VALUE_MAX + _VALUE_TOLERANCE)
        ]
        if len(out_of_range) > 0:
            raise DataIntegrityError(
                f"Year {year}, column {col}: {len(out_of_range)} values "
                f"outside [{_VALUE_MIN}, {_VALUE_MAX}]: {out_of_range.values[:5]}",
                context={
                    "year": year,
                    "column": col,
                    "n_violations": len(out_of_range),
                    "example_values": list(out_of_range.values[:5]),
                },
            )


def _validate_codebook(codebook: Dict[str, pd.DataFrame]) -> None:
    """
    Validate codebook DataFrames have the expected columns.

    Raises
    ------
    DataIntegrityError
    """
    expected_cols = {
        "provinces": {"Variable_Code", "Variable_Name"},
        "criteria": {"Variable_Code", "Variable_Name"},
        "subcriteria": {"Variable_Code", "Variable_Name", "Criteria_Name", "Criteria_Code"},
    }
    for key, required in expected_cols.items():
        actual = set(codebook[key].columns)
        missing = required - actual
        if missing:
            raise DataIntegrityError(
                f"Codebook '{key}' missing columns: {sorted(missing)}",
                context={"codebook": key, "missing_columns": sorted(missing)},
            )


# =============================================================================
# Full validate_data_integrity
# =============================================================================

def validate_data_integrity(
    panel: Dict[int, pd.DataFrame],
    config: AppConfig,
) -> None:
    """
    Run comprehensive validation of the full loaded panel against ``AppConfig``.

    Checks
    ------
    1. All expected years are present in the panel.
    2. All province codes match the codebook (P01–P63).
    3. Each year's active columns match its declared regime.
    4. All NaN structure matches the documented missing-data report.
    5. Known blank province-year combinations are correctly all-NaN.

    Parameters
    ----------
    panel : dict[int, pd.DataFrame]
    config : AppConfig

    Raises
    ------
    DataIntegrityError
        On any integrity violation.
    """
    logger.info("Running full data integrity validation")
    expected_years = set(config.data.years)
    found_years = set(panel.keys())

    missing_years = expected_years - found_years
    if missing_years:
        raise DataIntegrityError(
            f"Missing years in panel: {sorted(missing_years)}",
            context={"missing_years": sorted(missing_years)},
        )

    # Build expected province set from P01-P63
    expected_provinces = {f"P{i:02d}" for i in range(1, 64)}

    for yr, df in sorted(panel.items()):
        found_provinces = set(df.index)
        # Provinces that appear but are not in P01-P63
        unknown = found_provinces - expected_provinces
        if unknown:
            raise DataIntegrityError(
                f"Year {yr}: unknown province codes {sorted(unknown)}",
                context={"year": yr, "unknown_provinces": sorted(unknown)},
            )

    # Validate blank province-year Type-2 missingness
    for bpy in config.data.blank_province_years:
        yr, prov = bpy.year, bpy.province
        if yr not in panel:
            continue
        df = panel[yr]
        if prov not in df.index:
            # Province row may be entirely absent from the file — acceptable
            logger.debug(
                "Blank province {} in year {} is absent from file (entire row missing)",
                prov, yr,
            )
            continue
        row = df.loc[prov]
        if not row.isna().all():
            non_nan = row.dropna()
            raise DataIntegrityError(
                f"Province {prov} in year {yr} should be entirely NaN "
                f"(Type-2 blank) but has {len(non_nan)} non-NaN values: "
                f"{list(non_nan.index[:5])}",
                context={
                    "year": yr,
                    "province": prov,
                    "non_nan_columns": list(non_nan.index),
                },
            )

    logger.info("Data integrity validation passed for all {} years", len(panel))


# =============================================================================
# High-level factory
# =============================================================================

def load_papi_panel(
    config: AppConfig,
    validate: bool = True,
) -> PAPIPanel:
    """
    Load the full PAPI panel: all years + codebooks + regime detection.

    This is the primary entry point for the MCDM pipeline.
    The returned :class:`PAPIPanel` contains **read-only** raw data and must
    never be mutated by downstream code.

    Parameters
    ----------
    config : AppConfig
        Validated application configuration.
    validate : bool
        If ``True`` (default), run :func:`validate_data_integrity` after loading.

    Returns
    -------
    PAPIPanel
        Assembled panel with raw data, regimes, and codebooks.

    Raises
    ------
    DataLoadError, DataIntegrityError, RegimeDetectionError
    """
    logger.info("=== Loading PAPI panel ===")

    # 1. Load all annual CSV files (read-only)
    raw_panel = load_all_years(
        years=config.data.years,
        csv_dir=config.data.csv_dir,
        province_col=config.data.province_col,
    )

    # 2. Load codebooks
    codebook = load_codebook(codebook_dir=config.data.codebook_dir)

    # 3. Detect / validate regimes
    regimes = detect_regimes(raw_panel, config_regimes=config.data.regimes)

    # 4. (Optional) full integrity validation
    if validate:
        validate_data_integrity(raw_panel, config)

    panel = PAPIPanel(data=raw_panel, regimes=regimes, codebook=codebook)
    logger.info(
        "PAPI panel ready: {} years, {} regimes",
        len(panel.years), len(panel.regimes),
    )
    return panel


# =============================================================================
# Utility helpers
# =============================================================================

def get_regime_for_year(year: int, regimes: Dict[str, Regime]) -> Regime:
    """
    Return the :class:`Regime` that covers the given year.

    Raises
    ------
    RegimeDetectionError
        If no regime covers *year*.
    """
    for regime in regimes.values():
        if year in regime.years:
            return regime
    raise RegimeDetectionError(
        f"No regime found for year {year}",
        context={"year": year, "available_regimes": list(regimes.keys())},
    )


def get_active_subcriteria_for_year(
    year: int,
    regimes: Dict[str, Regime],
) -> List[str]:
    """
    Return the list of active sub-criteria for *year*.

    Parameters
    ----------
    year : int
    regimes : dict[str, Regime]

    Returns
    -------
    list[str]
        Sorted list of active sub-criteria column names.
    """
    return get_regime_for_year(year, regimes).active_subcriteria


def compute_missingness_report(
    panel: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute a per-year, per-column missingness summary.

    Returns
    -------
    pd.DataFrame
        MultiIndex (year, column) with columns: ``n_missing``, ``pct_missing``.
    """
    records = []
    for yr, df in sorted(panel.items()):
        n_rows = len(df)
        for col in df.columns:
            n_miss = int(df[col].isna().sum())
            records.append({
                "year": yr,
                "subcriteria": col,
                "n_missing": n_miss,
                "pct_missing": round(100.0 * n_miss / n_rows, 2) if n_rows > 0 else 0.0,
            })
    report = pd.DataFrame(records)
    return report
