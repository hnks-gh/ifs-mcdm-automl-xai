"""
src/mcdm/analysis/ranking_validation.py
----------------------------------------
Ranking Validation & Comparison Analysis.

This module provides comprehensive cross-method and temporal analysis of MCDM
rankings to validate their consistency and discriminatory power.

Three validation dimensions
===========================

**1. Inter-Method Agreement (Spearman's ρ)**
Measures correlation between rankings from different methods (IF-WASPAS,
IF-TOPSIS, IF-PROMETHEE II). High correlation indicates robust consensus;
low correlation suggests method-specific artifacts.

- Per-year ρ: correlation within each year
- Overall ρ: correlation across all years (treating all province-year pairs)
- Pairwise ρ: one ρ per (method1, method2) pair per year

**2. Discriminatory Power (IQR of score values)**
Measures the spread of provinces' scores. Higher IQR indicates better
discrimination (more granular differentiation). Computed from raw score values
S(x) = μ - ν for each IFS, not the final ranks.

- Per-method, per-year IQR
- Average IQR across years
- Method-level IQR comparison

**3. Temporal Persistence (Year-to-Year Spearman ρ)**
Measures how stable province rankings are across consecutive years within each
method. High persistence suggests stable governance patterns; low persistence
suggests volatile indicators.

- Year-to-year ρ per method
- Average ρ across all consecutive pairs
- Trend analysis (ρ improving/degrading over time)

References
----------
Spearman rank correlation: non-parametric measure of monotonic relationship.
    ρ = 1 - (6·Σd²) / (n·(n²-1)), where d = rank_difference
    Robust to outliers; undefined for tied ranks.

IQR (Interquartile Range): Q3 - Q1 = 75th percentile - 25th percentile.
    Unit-free measure of spread; robust to outliers.

Data Integrity Requirements
============================
- All ranks must be permutations of 1..n (no gaps, no ties for primary ranking)
- All provinces present in all methods for a given year
- Scores must be interpretable (typically S(x) ∈ [-1, 1] for IFS)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.core.schema import RankingMethod, RankingResult
from src.core.exceptions import FrameworkError

logger = logging.getLogger(__name__)

_TOL: float = 1e-12


@dataclass
class RankingValidationResult:
    """
    Complete ranking validation analysis output.

    Attributes
    ----------
    inter_method_correlation : dict[str, dict[str, float]]
        Spearman ρ between each pair of methods.
        Structure: {method1: {method2: rho_value}}
    inter_method_correlation_overall : dict[str, float]
        Spearman ρ per method pair, aggregated across all years.
    discriminatory_power_iqr : dict[RankingMethod, dict[int, float]]
        IQR of scores per method per year.
    discriminatory_power_mean : dict[RankingMethod, float]
        Average IQR across all years per method.
    temporal_persistence_rho : dict[RankingMethod, list[float]]
        Year-to-year Spearman ρ for each consecutive year pair.
    temporal_persistence_mean : dict[RankingMethod, float]
        Average year-to-year ρ per method.
    temporal_persistence_trend : dict[RankingMethod, Optional[float]]
        Linear trend (slope) of year-to-year ρ over time (None if < 2 points).
    """
    inter_method_correlation: Dict[str, Dict[str, float]]
    inter_method_correlation_overall: Dict[str, float]
    discriminatory_power_iqr: Dict[str, Dict[int, float]]
    discriminatory_power_mean: Dict[str, float]
    temporal_persistence_rho: Dict[str, List[float]]
    temporal_persistence_mean: Dict[str, float]
    temporal_persistence_trend: Dict[str, Optional[float]]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "inter_method_correlation": self.inter_method_correlation,
            "inter_method_correlation_overall": self.inter_method_correlation_overall,
            "discriminatory_power_iqr": {
                k: {str(y): v for y, v in vv.items()}
                for k, vv in self.discriminatory_power_iqr.items()
            },
            "discriminatory_power_mean": self.discriminatory_power_mean,
            "temporal_persistence_rho": {k: v for k, v in self.temporal_persistence_rho.items()},
            "temporal_persistence_mean": self.temporal_persistence_mean,
            "temporal_persistence_trend": self.temporal_persistence_trend,
        }

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Convert to pandas DataFrames for export.

        Returns
        -------
        inter_method_df : pd.DataFrame
            Inter-method Spearman ρ correlation matrix.
        discriminatory_df : pd.DataFrame
            IQR per method per year.
        temporal_df : pd.DataFrame
            Year-to-year ρ and trend per method.
        """
        # Inter-method correlation matrix (symmetric)
        methods_list = list(self.inter_method_correlation.keys())
        inter_method_data = {
            m1: [self.inter_method_correlation.get(m1, {}).get(m2, np.nan) for m2 in methods_list]
            for m1 in methods_list
        }
        inter_method_df = pd.DataFrame(inter_method_data, index=methods_list)

        # Discriminatory power dataframe
        discriminatory_df = pd.DataFrame(self.discriminatory_power_iqr).T

        # Temporal persistence dataframe
        temporal_data = []
        for method, rho_vals in self.temporal_persistence_rho.items():
            temporal_data.append({
                "Method": method,
                "Mean_RhoYoY": self.temporal_persistence_mean[method],
                "Trend": self.temporal_persistence_trend.get(method),
                "N_YearPairs": len(rho_vals),
            })
        temporal_df = pd.DataFrame(temporal_data)

        return inter_method_df, discriminatory_df, temporal_df


# =============================================================================
# Individual metric functions
# =============================================================================

def compute_spearman_rho(
    rank1: List[int],
    rank2: List[int],
) -> float:
    """
    Compute Spearman's rank correlation coefficient.

    Parameters
    ----------
    rank1, rank2 : list[int]
        Two rankings (typically permutations of 1..n).

    Returns
    -------
    rho : float
        Spearman's ρ ∈ [-1, 1].
    """
    if len(rank1) != len(rank2):
        raise ValueError(f"Rankings have different lengths: {len(rank1)} vs {len(rank2)}")
    if len(rank1) < 2:
        raise ValueError(f"Need at least 2 data points, got {len(rank1)}")

    rho, _ = spearmanr(rank1, rank2)
    if np.isnan(rho):
        return 0.0  # fallback for degenerate cases (all identical ranks)
    return float(rho)


def compute_score_iqr(
    scores: np.ndarray,
) -> float:
    """
    Compute Interquartile Range (IQR) of scores.

    Parameters
    ----------
    scores : ndarray, shape (n,)
        Score values (typically S(x) = μ - ν).

    Returns
    -------
    iqr : float
        Q3 - Q1 = 75th percentile - 25th percentile.
    """
    if len(scores) < 4:
        logger.warning(f"IQR with < 4 points may be unstable; got {len(scores)}")

    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr_val = q3 - q1

    return float(iqr_val)


def linear_trend(
    y_values: List[float],
) -> Optional[float]:
    """
    Compute linear trend (slope) of a time series.

    Parameters
    ----------
    y_values : list[float]
        Values to fit.

    Returns
    -------
    slope : float or None
        Linear regression slope. None if < 2 points.
    """
    if len(y_values) < 2:
        return None

    x = np.arange(len(y_values))
    y = np.array(y_values)

    # Remove NaN values
    valid_mask = ~np.isnan(y)
    if np.sum(valid_mask) < 2:
        return None

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Fit y = a + b*x; return b (slope)
    coeffs = np.polyfit(x_valid, y_valid, 1)
    return float(coeffs[0])


# =============================================================================
# Aggregated analysis functions
# =============================================================================

def compute_inter_method_agreement(
    rankings_per_year: Dict[int, Dict[RankingMethod, RankingResult]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute Spearman ρ between all pairs of ranking methods.

    Parameters
    ----------
    rankings_per_year : dict[int, dict[RankingMethod, RankingResult]]
        Rankings per year per method.

    Returns
    -------
    per_year_rho : dict[str, dict[str, float]]
        Spearman ρ per year. Structure: {year: {method_pair: rho}}
    overall_rho : dict[str, float]
        Spearman ρ across all years per method pair.
        Structure: {method_pair: rho}
    """
    if not rankings_per_year:
        raise ValueError("rankings_per_year is empty")

    years = sorted(rankings_per_year.keys())
    method_list = list(rankings_per_year[years[0]].keys())

    per_year_rho = {}
    all_ranks_pairs = {
        (m1, m2): ([], [])
        for m1 in method_list
        for m2 in method_list
        if m1 < m2  # Avoid duplicate pairs
    }

    for year in years:
        year_rankings = rankings_per_year[year]
        year_rho = {}

        for m1 in method_list:
            for m2 in method_list:
                if m1 < m2:
                    rho = compute_spearman_rho(
                        year_rankings[m1].ranks,
                        year_rankings[m2].ranks,
                    )
                    pair_key = f"{m1.value}_vs_{m2.value}"
                    year_rho[pair_key] = rho

                    # Accumulate for overall ρ
                    all_ranks_pairs[(m1, m2)][0].extend(year_rankings[m1].ranks)
                    all_ranks_pairs[(m1, m2)][1].extend(year_rankings[m2].ranks)

        per_year_rho[year] = year_rho

    # Compute overall ρ across all years
    overall_rho = {}
    for (m1, m2), (ranks1, ranks2) in all_ranks_pairs.items():
        if ranks1:  # Check not empty
            rho_overall = compute_spearman_rho(ranks1, ranks2)
            pair_key = f"{m1.value}_vs_{m2.value}"
            overall_rho[pair_key] = rho_overall

    return per_year_rho, overall_rho


def compute_discriminatory_power(
    rankings_per_year: Dict[int, Dict[RankingMethod, RankingResult]],
) -> Tuple[Dict[RankingMethod, Dict[int, float]], Dict[RankingMethod, float]]:
    """
    Compute discriminatory power (IQR of scores) per method per year.

    Parameters
    ----------
    rankings_per_year : dict[int, dict[RankingMethod, RankingResult]]
        Rankings per year per method (with score values).

    Returns
    -------
    iqr_per_year : dict[RankingMethod, dict[int, float]]
        IQR per method per year.
    iqr_mean : dict[RankingMethod, float]
        Average IQR per method across all years.
    """
    if not rankings_per_year:
        raise ValueError("rankings_per_year is empty")

    years = sorted(rankings_per_year.keys())
    method_list = list(rankings_per_year[years[0]].keys())

    iqr_per_year = {method: {} for method in method_list}
    iqr_mean = {}

    for method in method_list:
        method_iqr_values = []
        for year in years:
            result = rankings_per_year[year][method]
            scores = np.array(result.scores)
            iqr_val = compute_score_iqr(scores)
            iqr_per_year[method][year] = iqr_val
            method_iqr_values.append(iqr_val)

        iqr_mean[method] = float(np.mean(method_iqr_values))

    return iqr_per_year, iqr_mean


def compute_temporal_persistence(
    rankings_per_year: Dict[int, Dict[RankingMethod, RankingResult]],
) -> Tuple[Dict[RankingMethod, List[float]], Dict[RankingMethod, float], Dict[RankingMethod, Optional[float]]]:
    """
    Compute year-to-year ranking persistence (Spearman ρ).

    Parameters
    ----------
    rankings_per_year : dict[int, dict[RankingMethod, RankingResult]]
        Rankings per year per method.

    Returns
    -------
    rho_yoy : dict[RankingMethod, list[float]]
        Year-to-year Spearman ρ for each consecutive pair.
    rho_mean : dict[RankingMethod, float]
        Average year-to-year ρ per method.
    trend : dict[RankingMethod, Optional[float]]
        Linear trend (slope) of year-to-year ρ over time.
    """
    if not rankings_per_year:
        raise ValueError("rankings_per_year is empty")

    years = sorted(rankings_per_year.keys())
    if len(years) < 2:
        raise ValueError("Need at least 2 consecutive years for temporal persistence")

    method_list = list(rankings_per_year[years[0]].keys())

    rho_yoy = {method: [] for method in method_list}
    rho_mean = {}
    trend = {}

    for method in method_list:
        for i in range(len(years) - 1):
            year_t = years[i]
            year_t1 = years[i + 1]

            result_t = rankings_per_year[year_t][method]
            result_t1 = rankings_per_year[year_t1][method]

            rho = compute_spearman_rho(result_t.ranks, result_t1.ranks)
            rho_yoy[method].append(rho)

        rho_mean[method] = float(np.mean(rho_yoy[method]))
        trend[method] = linear_trend(rho_yoy[method])

    return rho_yoy, rho_mean, trend


# =============================================================================
# Main validation function
# =============================================================================

def run_ranking_validation(
    rankings_per_year: Dict[int, Dict[RankingMethod, RankingResult]],
) -> RankingValidationResult:
    """
    Execute comprehensive ranking validation analysis.

    Parameters
    ----------
    rankings_per_year : dict[int, dict[RankingMethod, RankingResult]]
        Rankings per year per method.
        Structure: {year: {RankingMethod: RankingResult}}

    Returns
    -------
    result : RankingValidationResult
        Complete validation analysis.

    Raises
    ------
    FrameworkError
        If rankings are invalid or computation fails.
    """
    logger.info("Running ranking validation analysis...")

    try:
        # Inter-method agreement
        logger.info("Computing inter-method agreement (Spearman ρ)...")
        per_year_rho, overall_rho = compute_inter_method_agreement(rankings_per_year)

        # Flatten per-year results to nested dict structure
        inter_method_by_year = {}
        for year, rho_dict in per_year_rho.items():
            for pair_key, rho_val in rho_dict.items():
                if pair_key not in inter_method_by_year:
                    inter_method_by_year[pair_key] = {}
                inter_method_by_year[pair_key][year] = rho_val

        # Discriminatory power
        logger.info("Computing discriminatory power (IQR of scores)...")
        iqr_per_year, iqr_mean = compute_discriminatory_power(rankings_per_year)

        # Temporal persistence
        logger.info("Computing temporal persistence (year-to-year ρ)...")
        rho_yoy, rho_mean, trend = compute_temporal_persistence(rankings_per_year)

        # Log summaries
        logger.info(f"Inter-method agreement (overall): {overall_rho}")
        logger.info(f"Discriminatory power (mean IQR): {iqr_mean}")
        logger.info(f"Temporal persistence (mean ρ_YoY): {rho_mean}")

        result = RankingValidationResult(
            inter_method_correlation=inter_method_by_year,
            inter_method_correlation_overall=overall_rho,
            discriminatory_power_iqr={k.value: v for k, v in iqr_per_year.items()},
            discriminatory_power_mean={k.value: v for k, v in iqr_mean.items()},
            temporal_persistence_rho={k.value: v for k, v in rho_yoy.items()},
            temporal_persistence_mean={k.value: v for k, v in rho_mean.items()},
            temporal_persistence_trend={k.value: v for k, v in trend.items()},
        )

        logger.info("Ranking validation analysis complete")
        return result

    except Exception as e:
        raise FrameworkError(
            f"Ranking validation failed: {str(e)}",
            context={"error": str(e)},
        ) from e


def save_ranking_validation(
    result: RankingValidationResult,
    output_dir: str,
) -> None:
    """
    Save ranking validation results to disk.

    Parameters
    ----------
    result : RankingValidationResult
        Validation result.
    output_dir : str
        Directory to save outputs.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inter_method_df, discriminatory_df, temporal_df = result.to_dataframe()

    # Save inter-method correlation
    inter_method_df.to_csv(output_dir / "ranking_validation_inter_method_spearman.csv")
    logger.info(f"Saved inter-method correlations to {output_dir / 'ranking_validation_inter_method_spearman.csv'}")

    # Save discriminatory power
    discriminatory_df.to_csv(output_dir / "ranking_validation_discriminatory_power_iqr.csv")
    logger.info(f"Saved discriminatory power to {output_dir / 'ranking_validation_discriminatory_power_iqr.csv'}")

    # Save temporal persistence
    temporal_df.to_csv(output_dir / "ranking_validation_temporal_persistence_yoy.csv", index=False)
    logger.info(f"Saved temporal persistence to {output_dir / 'ranking_validation_temporal_persistence_yoy.csv'}")

    # Save overall correlation summary
    overall_corr_df = pd.DataFrame(
        list(result.inter_method_correlation_overall.items()),
        columns=["Method_Pair", "Spearman_Rho"],
    )
    overall_corr_df.to_csv(output_dir / "ranking_validation_inter_method_overall.csv", index=False)
    logger.info(f"Saved overall correlations to {output_dir / 'ranking_validation_inter_method_overall.csv'}")

    # Save detailed JSON for reproducibility
    import json
    with open(output_dir / "ranking_validation_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Saved detailed result to {output_dir / 'ranking_validation_result.json'}")
