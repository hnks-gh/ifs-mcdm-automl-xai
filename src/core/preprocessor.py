"""
src/core/preprocessor.py
------------------------
Data preprocessing utilities that bridge raw PAPI CSV data and the IFS
arithmetic layer.

Responsibilities
----------------
1. Normalise raw scores (optional pre-IFS step).
2. Apply regime mask: restrict each year's DataFrame to its active sub-criteria.
3. Complete-case exclusion: remove province rows that are entirely NaN.
4. Convert the full PAPI panel to a dict of IFSMatrix objects.

Data isolation guarantee
------------------------
ALL functions here operate on *copies* of the input DataFrames — the original
``data/csv/`` files and the ``PAPIPanel.data`` dict are NEVER mutated.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.exceptions import DataIntegrityError
from src.core.ifs_arithmetic import IFSMatrix, ifs_matrix_from_dataframe
from src.core.schema import AppConfig, IFSConfig, Regime
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# 1. Normalise raw scores
# =============================================================================

def normalize_raw_scores(
    df: pd.DataFrame,
    method: str = "theoretical_max",
    x_max: float = 3.33,
    col_maxima: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Normalise a province × sub-criteria DataFrame of raw scores to [0, 1].

    This function is **not** required before IFS conversion because
    ``vec_score_to_ifs`` performs the normalisation internally.  It is
    provided as a utility for exploratory analysis and alternative pipelines.

    Parameters
    ----------
    df : pd.DataFrame
        Province-indexed raw scores (NaN allowed).
    method : str
        ``"theoretical_max"`` — divide by ``x_max`` (3.33 for PAPI).
        ``"max_observed"``    — divide each column by its maximum across all
                                available data (uses ``col_maxima`` if supplied,
                                otherwise computed from ``df``).
        ``"minmax"``          — per-column min-max to [0, 1].
    x_max : float
        Theoretical maximum; used only when ``method="theoretical_max"``.
    col_maxima : dict[str, float] | None
        Pre-computed per-column maxima.  When ``method="max_observed"`` and
        this is None, maxima are computed from ``df`` itself (use with caution:
        train/test split awareness is the caller's responsibility).

    Returns
    -------
    pd.DataFrame
        New DataFrame of the same shape, values in [0, 1], NaN preserved.

    Raises
    ------
    DataIntegrityError
        If an unknown method is requested.
    """
    df = df.copy()

    if method == "theoretical_max":
        df = df / x_max

    elif method == "max_observed":
        if col_maxima is not None:
            for col in df.columns:
                if col in col_maxima and col_maxima[col] > 0:
                    df[col] = df[col] / col_maxima[col]
        else:
            for col in df.columns:
                col_max = df[col].max()
                if pd.notna(col_max) and col_max > 0:
                    df[col] = df[col] / col_max

    elif method == "minmax":
        for col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            rng = col_max - col_min
            if pd.notna(rng) and rng > 0:
                df[col] = (df[col] - col_min) / rng
            else:
                # Zero range → set to 0.5 (neutral)
                df[col] = 0.5

    else:
        raise DataIntegrityError(
            f"Unknown normalisation method: '{method}'",
            context={"method": method},
        )

    # Clip to [0, 1] to catch floating-point boundary violations
    df = df.clip(lower=0.0, upper=1.0)
    return df


# =============================================================================
# 2. Apply regime mask
# =============================================================================

def apply_regime_mask(
    df: pd.DataFrame,
    regime: Regime,
    fill_absent: float = float("nan"),
) -> pd.DataFrame:
    """
    Return a copy of *df* restricted to the regime's active sub-criteria.

    Columns absent in the regime are set to ``fill_absent`` (NaN by default)
    in the returned DataFrame so that vectorised downstream code can detect and
    skip them.  Province rows and index are preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Province-indexed DataFrame containing some or all 29 sub-criteria.
    regime : Regime
        The regime whose ``active_subcriteria`` defines the allowed columns.
    fill_absent : float
        Value inserted for absent columns.  Default is NaN (recommended).

    Returns
    -------
    pd.DataFrame
        Same columns as ``df``, but absent-regime columns are filled with
        ``fill_absent``.
    """
    df = df.copy()
    absent = set(regime.absent_subcriteria)
    for col in df.columns:
        if col in absent:
            df[col] = fill_absent
    return df


# =============================================================================
# 3. Complete-case exclusion
# =============================================================================

def complete_case_exclusion(
    df: pd.DataFrame,
    active_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Remove province rows that are entirely NaN across all active sub-criteria.

    These correspond to Type-2 missing observations (entire province-year
    blank) documented in the PAPI missing data report.  The remaining
    province rows are returned unchanged — partial missingness (Type 3) is
    handled downstream in the CRITIC computation by weight renormalisation.

    Parameters
    ----------
    df : pd.DataFrame
        Province-indexed DataFrame (NaN allowed).
    active_cols : list[str] | None
        Subset of columns to consider for the all-NaN check.
        If None, all columns in ``df`` are used.

    Returns
    -------
    pd.DataFrame
        DataFrame with entirely-blank province rows removed.
    """
    cols = active_cols if active_cols is not None else list(df.columns)
    # Keep only columns that exist in df
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df.copy()

    all_nan_mask = df[cols].isna().all(axis=1)
    n_dropped = int(all_nan_mask.sum())
    if n_dropped > 0:
        dropped = list(df.index[all_nan_mask])
        logger.debug(
            "complete_case_exclusion: dropped {} all-NaN province rows: {}",
            n_dropped, dropped,
        )
    return df[~all_nan_mask].copy()


# =============================================================================
# 4. Convert full panel to IFS
# =============================================================================

def convert_panel_to_ifs(
    panel: Dict[int, pd.DataFrame],
    regimes: Dict[str, Regime],
    ifs_config: IFSConfig,
) -> Dict[int, IFSMatrix]:
    """
    Convert the full PAPI raw-score panel to IFS matrices.

    For each year:
    1. Identify the regime → get active sub-criteria.
    2. Apply regime mask (absent cols → NaN).
    3. Perform complete-case exclusion (drop all-NaN province rows).
    4. Convert remaining raw scores to IFS triples via ``vec_score_to_ifs``.

    The returned dict is completely independent of the input panel dict;
    no in-place modification is performed.

    Parameters
    ----------
    panel : dict[int, pd.DataFrame]
        Raw PAPI panel (output of :func:`src.core.data_loader.load_all_years`).
    regimes : dict[str, Regime]
        Regime metadata (output of :func:`src.core.data_loader.detect_regimes`).
    ifs_config : IFSConfig
        IFS conversion configuration (``score_max``, ``fixed_pi_value``).

    Returns
    -------
    dict[int, IFSMatrix]
        Mapping year → IFSMatrix, restricted to active sub-criteria and
        valid (non-blank) province rows.
    """
    from src.core.data_loader import get_regime_for_year  # avoid circular at top level

    ifs_panel: Dict[int, IFSMatrix] = {}
    x_max = ifs_config.score_max
    pi_fixed = ifs_config.fixed_pi_value

    for year in sorted(panel.keys()):
        df_raw = panel[year].copy()

        # Step 1: get regime
        try:
            regime = get_regime_for_year(year, regimes)
        except Exception as exc:
            logger.error("Cannot find regime for year {}: {}", year, exc)
            raise

        # Step 2: apply regime mask (absent → NaN)
        df_masked = apply_regime_mask(df_raw, regime, fill_absent=float("nan"))

        # Step 3: complete-case exclusion over active sub-criteria
        df_clean = complete_case_exclusion(df_masked, active_cols=regime.active_subcriteria)

        # Step 4: select only active columns (preserve order from codebook)
        active_in_df = [c for c in regime.active_subcriteria if c in df_clean.columns]
        df_active = df_clean[active_in_df]

        # Step 5: build IFSMatrix
        ifs_mat = ifs_matrix_from_dataframe(
            df=df_active,
            x_max=x_max,
            pi_fixed=pi_fixed,
            year=year,
        )

        ifs_panel[year] = ifs_mat
        logger.debug(
            "Year {} → IFSMatrix shape ({}, {})  regime={}",
            year, ifs_mat.n_alternatives, ifs_mat.n_criteria, regime.regime_id,
        )

    logger.info(
        "convert_panel_to_ifs: {} years converted to IFSMatrix objects", len(ifs_panel)
    )
    return ifs_panel


# =============================================================================
# Utility: compute per-column observed maxima across all years
# =============================================================================

def compute_panel_maxima(panel: Dict[int, pd.DataFrame]) -> Dict[str, float]:
    """
    Compute the maximum observed value per sub-criterion across all years.

    Useful for ``normalize_raw_scores(method='max_observed')`` when a
    consistent cross-year maximum is required.

    Parameters
    ----------
    panel : dict[int, pd.DataFrame]

    Returns
    -------
    dict[str, float]
        Mapping sub-criterion code → maximum value observed across all years.
    """
    all_cols: Dict[str, float] = {}
    for df in panel.values():
        for col in df.columns:
            col_max = float(df[col].max()) if not df[col].isna().all() else 0.0
            if col not in all_cols or col_max > all_cols[col]:
                all_cols[col] = col_max
    return all_cols
