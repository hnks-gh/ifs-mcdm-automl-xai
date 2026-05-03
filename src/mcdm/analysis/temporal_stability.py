"""
src/mcdm/analysis/temporal_stability.py
----------------------------------------
Temporal Stability Analysis for IF-CRITIC Weights.

Temporal stability quantifies how much the IF-CRITIC weight vectors fluctuate
across overlapping time windows. This analysis:

1. **Sliding window decomposition**: Partition 14 years (2011-2024) into 10
   overlapping windows of 5 years each.
   - Window 1: 2011-2015
   - Window 2: 2012-2016
   - ...
   - Window 10: 2020-2024

2. **Per-window weight computation**: For each window, compute IF-CRITIC weights
   using only the observations in that window (thus capturing era-specific patterns).

3. **Stability metrics**:
   - **RMSD (Root Mean Square Deviation)**: Euclidean distance between consecutive
     window weight vectors. Measures year-over-year volatility.
   - **CV (Coefficient of Variation)**: std(w_j across windows) / mean(w_j across windows).
     Measures relative variability per sub-criterion.

Mathematical Specification
===========================

RMSD between windows t and t+1:
    RMSD_t = sqrt(mean((w_t[j] - w_{t+1}[j])^2)) for j in 1..n_subcriteria

CV for sub-criterion j:
    CV_j = std(w[j] across all windows) / mean(w[j] across all windows)
    → nan if mean is near zero

Interpretation
==============
- Low RMSD/CV: weights are stable; methods are robust to temporal aggregation
- High RMSD/CV: weights fluctuate; results may be sensitive to the analysis period
- RMSD trend: increasing RMSD suggests structural breaks in the data

References
----------
None specific; derived from standard signal processing and time-series analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.schema import RankingMethod, WeightVector
from src.core.exceptions import FrameworkError
from src.mcdm.weighting.two_level_aggregator import compute_weights_for_all_years

logger = logging.getLogger(__name__)

_TOL: float = 1e-12


@dataclass
class TemporalStabilityResult:
    """
    Output of temporal stability analysis.

    Attributes
    ----------
    window_years : list[tuple[int, int]]
        List of (start_year, end_year) for each window.
    window_weights : list[WeightVector]
        IF-CRITIC weights for each window, in order.
    rmsd_consecutive : list[float]
        RMSD between consecutive windows. Length = n_windows - 1.
    cv_per_subcriteria : dict[str, float]
        Coefficient of Variation for each sub-criterion.
    cv_overall : float
        Mean CV across all sub-criteria.
    rmsd_mean : float
        Mean RMSD across all consecutive window pairs.
    rmsd_std : float
        Standard deviation of RMSD values.
    """
    window_years: List[Tuple[int, int]]
    window_weights: List[WeightVector]
    rmsd_consecutive: List[float]
    cv_per_subcriteria: Dict[str, float]
    cv_overall: float
    rmsd_mean: float
    rmsd_std: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "window_years": self.window_years,
            "window_weights": [w.values for w in self.window_weights],
            "window_labels": [w.labels for w in self.window_weights],
            "rmsd_consecutive": self.rmsd_consecutive,
            "cv_per_subcriteria": self.cv_per_subcriteria,
            "cv_overall": self.cv_overall,
            "rmsd_mean": self.rmsd_mean,
            "rmsd_std": self.rmsd_std,
        }

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert to pandas DataFrames for easy export.

        Returns
        -------
        weights_df : pd.DataFrame
            Shape (n_windows, n_subcriteria) - weights per window.
        metrics_df : pd.DataFrame
            Shape (2 + n_windows-1, 1) - RMSD and CV metrics.
        """
        # Extract weight labels from first window
        labels = self.window_weights[0].labels if self.window_weights else []

        # Build weights dataframe
        weight_arrays = [w.values for w in self.window_weights]
        weights_df = pd.DataFrame(
            weight_arrays,
            columns=labels,
            index=[f"W{i+1}_{y1}-{y2}" for i, (y1, y2) in enumerate(self.window_years)],
        )

        # Build metrics dataframe
        metrics_data = {}
        metrics_data["RMSD (consecutive windows)"] = (
            self.rmsd_consecutive + [np.nan] * (len(self.window_years) - len(self.rmsd_consecutive))
        )
        metrics_df = pd.DataFrame(metrics_data)

        return weights_df, metrics_df


# =============================================================================
# Window generation
# =============================================================================

def generate_windows(
    years: List[int],
    window_size: int = 5,
    n_windows: Optional[int] = None,
) -> List[List[int]]:
    """
    Generate overlapping time windows from a list of years.

    Parameters
    ----------
    years : list[int]
        Sorted list of years, e.g. [2011, 2012, ..., 2024].
    window_size : int
        Size of each window (number of consecutive years).
    n_windows : int, optional
        Number of windows to generate. If None, generates maximum possible
        (len(years) - window_size + 1).

    Returns
    -------
    windows : list[list[int]]
        List of windows, each a list of consecutive years.
        Windows are in chronological order.

    Raises
    ------
    FrameworkError
        If window_size > len(years) or n_windows is invalid.
    """
    if not isinstance(years, list) or not years:
        raise ValueError("years must be a non-empty list")
    if not all(isinstance(y, int) for y in years):
        raise TypeError("All elements in years must be integers")
    if years != sorted(years):
        raise ValueError("years must be sorted in ascending order")

    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if window_size > len(years):
        raise FrameworkError(
            f"window_size ({window_size}) > total years ({len(years)})",
            context={"window_size": window_size, "total_years": len(years)},
        )

    max_windows = len(years) - window_size + 1
    if n_windows is None:
        n_windows = max_windows
    elif n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    elif n_windows > max_windows:
        raise FrameworkError(
            f"n_windows ({n_windows}) > max possible ({max_windows})",
            context={"n_windows": n_windows, "max_windows": max_windows},
        )

    windows = []
    step = max(1, (max_windows - 1) // (n_windows - 1)) if n_windows > 1 else 0

    if n_windows == 1:
        windows.append(years[:window_size])
    else:
        for i in range(n_windows):
            start_idx = i * step
            if start_idx + window_size <= len(years):
                windows.append(years[start_idx : start_idx + window_size])

    return windows


# =============================================================================
# Metric computation
# =============================================================================

def compute_rmsd(w1: WeightVector, w2: WeightVector) -> float:
    """
    Compute Root Mean Square Deviation between two weight vectors.

    Parameters
    ----------
    w1, w2 : WeightVector
        Two weight vectors with identical labels.

    Returns
    -------
    rmsd : float
        sqrt(mean((w1[j] - w2[j])^2)).
    """
    if len(w1.values) != len(w2.values):
        raise ValueError(f"Weight vector lengths do not match: {len(w1.values)} vs {len(w2.values)}")

    # Map labels to ensure alignment
    if w1.labels != w2.labels:
        logger.warning(
            "Weight vector labels differ; attempting alignment. This may indicate data inconsistency."
        )

    diff = np.array(w1.values) - np.array(w2.values)
    rmsd_val = np.sqrt(np.mean(diff**2))
    return float(rmsd_val)


def compute_cv_per_subcriteria(
    weight_series: List[WeightVector],
) -> Dict[str, float]:
    """
    Compute Coefficient of Variation for each sub-criterion across windows.

    Parameters
    ----------
    weight_series : list[WeightVector]
        Weight vectors for each window, in order.

    Returns
    -------
    cv_dict : dict[str, float]
        CV for each sub-criterion label. Value is nan if mean is near-zero.
    """
    if not weight_series:
        raise ValueError("weight_series is empty")

    n_subcriteria = len(weight_series[0].values)
    labels = weight_series[0].labels

    # Stack weights into matrix: shape (n_windows, n_subcriteria)
    weight_matrix = np.array([w.values for w in weight_series])

    cv_dict = {}
    for j, label in enumerate(labels):
        weights_j = weight_matrix[:, j]
        mean_j = np.mean(weights_j)
        std_j = np.std(weights_j, ddof=1)  # sample std

        if abs(mean_j) < _TOL:
            cv_dict[label] = np.nan
        else:
            cv_dict[label] = float(std_j / mean_j)

    return cv_dict


# =============================================================================
# Main analysis function
# =============================================================================

def run_temporal_stability(
    panel_dict: Dict[int, "pd.DataFrame"],  # type: ignore[name-defined]
    weighting_config,  # WeightingConfig
    analysis_config,  # TemporalStabilityConfig
    mcdm_config,  # MCDMConfig (for regime handling, etc.)
) -> TemporalStabilityResult:
    """
    Execute full temporal stability analysis.

    Parameters
    ----------
    panel_dict : dict[int, pd.DataFrame]
        Raw PAPI panel: year -> DataFrame (provinces × sub-criteria).
    weighting_config : WeightingConfig
        Weighting algorithm configuration.
    analysis_config : TemporalStabilityConfig
        Window size and other stability analysis parameters.
    mcdm_config : MCDMConfig
        Full MCDM configuration (includes regime definitions, etc.).

    Returns
    -------
    result : TemporalStabilityResult
        Complete stability analysis output.

    Raises
    ------
    FrameworkError
        If window generation or weight computation fails.
    """
    # Generate overlapping windows
    all_years = sorted(panel_dict.keys())
    logger.info(
        f"Generating {analysis_config.n_windows} overlapping windows of size "
        f"{analysis_config.window_size} from {len(all_years)} years"
    )

    windows = generate_windows(
        all_years,
        window_size=analysis_config.window_size,
        n_windows=analysis_config.n_windows,
    )
    logger.info(f"Generated {len(windows)} windows: {windows}")

    # Compute weights for each window
    window_weights_list = []
    window_years_tuples = []

    for i, window_years in enumerate(windows):
        logger.info(f"Computing weights for window {i+1}/{len(windows)}: years {window_years}")

        # Create sub-panel for this window
        window_panel = {year: panel_dict[year] for year in window_years}

        try:
            # Compute IF-CRITIC weights for this window
            weights_per_year = compute_weights_for_all_years(
                window_panel,
                weighting_config,
                mcdm_config,
            )

            # Average weights across the window (or select representative year)
            # Strategy: compute grand mean across all years in window
            all_weights_in_window = list(weights_per_year.values())
            averaged_weights_values = np.mean(
                [w.values for w in all_weights_in_window],
                axis=0,
            )

            # Create averaged weight vector
            labels = all_weights_in_window[0].labels
            avg_weight_vector = WeightVector(
                labels=labels,
                values=averaged_weights_values.tolist(),
            )
            window_weights_list.append(avg_weight_vector)
            window_years_tuples.append((window_years[0], window_years[-1]))

        except Exception as e:
            raise FrameworkError(
                f"Failed to compute weights for window {window_years}: {str(e)}",
                context={"window_years": window_years, "error": str(e)},
            ) from e

    logger.info(f"Successfully computed weights for all {len(windows)} windows")

    # Compute RMSD between consecutive windows
    rmsd_values = []
    for i in range(len(window_weights_list) - 1):
        rmsd_i = compute_rmsd(window_weights_list[i], window_weights_list[i + 1])
        rmsd_values.append(rmsd_i)
        logger.debug(f"RMSD(window {i+1}, window {i+2}) = {rmsd_i:.6f}")

    # Compute CV per sub-criterion
    cv_per_subcriteria = compute_cv_per_subcriteria(window_weights_list)
    cv_overall = float(np.nanmean(list(cv_per_subcriteria.values())))

    rmsd_mean = float(np.mean(rmsd_values)) if rmsd_values else np.nan
    rmsd_std = float(np.std(rmsd_values, ddof=1)) if len(rmsd_values) > 1 else np.nan

    logger.info(
        f"Temporal stability summary: RMSD_mean={rmsd_mean:.6f}, CV_overall={cv_overall:.6f}"
    )

    # Create result object
    result = TemporalStabilityResult(
        window_years=window_years_tuples,
        window_weights=window_weights_list,
        rmsd_consecutive=rmsd_values,
        cv_per_subcriteria=cv_per_subcriteria,
        cv_overall=cv_overall,
        rmsd_mean=rmsd_mean,
        rmsd_std=rmsd_std,
    )

    logger.info("Temporal stability analysis complete")
    return result


def save_temporal_stability(
    result: TemporalStabilityResult,
    output_dir: str,
) -> None:
    """
    Save temporal stability analysis results to disk.

    Parameters
    ----------
    result : TemporalStabilityResult
        Analysis result.
    output_dir : str
        Directory to save outputs.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save window weights
    weights_df, metrics_df = result.to_dataframe()
    weights_df.to_csv(output_dir / "temporal_stability_window_weights.csv")
    logger.info(f"Saved window weights to {output_dir / 'temporal_stability_window_weights.csv'}")

    # Save CV metrics
    cv_df = pd.DataFrame(
        list(result.cv_per_subcriteria.items()),
        columns=["Subcriteria", "CV"],
    )
    cv_df.to_csv(output_dir / "temporal_stability_cv_per_subcriteria.csv", index=False)
    logger.info(f"Saved CV metrics to {output_dir / 'temporal_stability_cv_per_subcriteria.csv'}")

    # Save RMSD values
    rmsd_df = pd.DataFrame(
        {
            "Window_Pair": [f"W{i+1}-W{i+2}" for i in range(len(result.rmsd_consecutive))],
            "RMSD": result.rmsd_consecutive,
        }
    )
    rmsd_df.to_csv(output_dir / "temporal_stability_rmsd_consecutive.csv", index=False)
    logger.info(f"Saved RMSD values to {output_dir / 'temporal_stability_rmsd_consecutive.csv'}")

    # Save summary statistics
    summary = pd.DataFrame(
        {
            "Metric": [
                "RMSD_Mean",
                "RMSD_Std",
                "CV_Overall",
                "N_Windows",
                "Window_Size",
            ],
            "Value": [
                result.rmsd_mean,
                result.rmsd_std,
                result.cv_overall,
                len(result.window_years),
                result.window_years[0][1] - result.window_years[0][0] + 1 if result.window_years else np.nan,
            ],
        }
    )
    summary.to_csv(output_dir / "temporal_stability_summary.csv", index=False)
    logger.info(f"Saved summary to {output_dir / 'temporal_stability_summary.csv'}")
