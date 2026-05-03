#!/usr/bin/env python
"""
scripts/03_mcdm_ranking_demo.py
-------------------------------
Demonstration of all three IF-MCDM ranking methods (WASPAS, TOPSIS, PROMETHEE II).

This script shows:
1. Loading and preparing PAPI data
2. Converting to IFS representation
3. Running all three ranking methods with the same weights
4. Comparing results and outputs

Run from project root:
    python scripts/03_mcdm_ranking_demo.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.core.data_loader import load_config, load_all_years
from src.core.preprocessor import normalize_raw_scores, complete_case_exclusion
from src.core.ifs_arithmetic import ifs_matrix_from_dataframe
from src.mcdm.ranking import if_waspas, if_topsis, if_promethee2
from src.utils.logger import get_logger
from src.utils.io_utils import save_ranking_results

logger = get_logger(__name__)


def main() -> None:
    """Run ranking demonstration."""
    logger.info("=" * 80)
    logger.info("Phase 4: MCDM Ranking Methods Demo")
    logger.info("=" * 80)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config("config/config.yaml")

    # Load PAPI panel data (2020 as example year)
    logger.info("Loading PAPI data for year 2020...")
    panel = load_all_years(config)
    year = 2020
    df_raw = panel.data[year]
    logger.info(f"  Loaded {len(df_raw)} provinces × {len(df_raw.columns)} sub-criteria")

    # Preprocess: normalize scores
    logger.info("Normalizing raw scores...")
    df_norm = normalize_raw_scores(df_raw, method="max_observed")

    # Handle missing data: complete-case exclusion
    logger.info("Applying complete-case exclusion for all-NaN rows...")
    df_clean = complete_case_exclusion(df_norm)
    logger.info(f"  After exclusion: {len(df_clean)} provinces")

    # Convert to IFS
    logger.info("Converting scores to Intuitionistic Fuzzy representation...")
    ifs_matrix = ifs_matrix_from_dataframe(
        df_clean,
        x_max=config.ifs.score_max,
        pi_fixed=config.ifs.fixed_pi_value,
        year=year,
    )
    logger.info(f"  IFS matrix: {ifs_matrix.n_alternatives} × {ifs_matrix.n_criteria}")

    # Create synthetic weights (equal) for demo purposes
    # In real usage, weights come from IF-CRITIC weighting phase
    weights = np.ones(ifs_matrix.n_criteria) / ifs_matrix.n_criteria
    logger.info(f"Using uniform weights for demo (n={len(weights)})")

    # Run all three ranking methods
    logger.info("\n" + "=" * 80)
    logger.info("Running ranking methods...")
    logger.info("=" * 80)

    # IF-WASPAS
    logger.info("\n[1/3] IF-WASPAS (Weighted Aggregated Sum Product Assessment)")
    logger.info("      Configuration: λ=0.5 (balanced WSM/WPM)")
    result_waspas = if_waspas.rank(
        ifs_matrix,
        weights,
        lambda_param=config.mcdm.ranking.if_waspas.lambda_param,
    )
    logger.info(f"      Top 5: {_format_top_n(result_waspas, 5)}")

    # IF-TOPSIS
    logger.info("\n[2/3] IF-TOPSIS (Technique for Order Preference by Similarity)")
    logger.info("      Configuration: distance=normalized Euclidean")
    result_topsis = if_topsis.rank(ifs_matrix, weights)
    logger.info(f"      Top 5: {_format_top_n(result_topsis, 5)}")

    # IF-PROMETHEE II
    logger.info("\n[3/3] IF-PROMETHEE II (Preference Ranking Organization Method)")
    logger.info(f"      Configuration: p={config.mcdm.ranking.if_promethee2.p_parameter}")
    result_promethee2 = if_promethee2.rank(
        ifs_matrix,
        weights,
        p_parameter=config.mcdm.ranking.if_promethee2.p_parameter,
    )
    logger.info(f"      Top 5: {_format_top_n(result_promethee2, 5)}")

    # Compare rankings
    logger.info("\n" + "=" * 80)
    logger.info("Ranking Comparison")
    logger.info("=" * 80)

    df_comparison = _create_comparison_df(result_waspas, result_topsis, result_promethee2)
    logger.info("\nTop 10 provinces by each method:")
    logger.info(df_comparison.head(10).to_string())

    # Inter-method agreement (Spearman correlation)
    logger.info("\n" + "=" * 80)
    logger.info("Inter-Method Agreement Analysis")
    logger.info("=" * 80)

    rank_arrays = {
        "WASPAS": result_waspas.ranks,
        "TOPSIS": result_topsis.ranks,
        "PROMETHEE II": result_promethee2.ranks,
    }

    from scipy.stats import spearmanr

    logger.info("\nSpearman Rank Correlation ρ:")
    logger.info("  (Measures agreement between ranking methods)")

    methods = list(rank_arrays.keys())
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                rho, pval = spearmanr(rank_arrays[m1], rank_arrays[m2])
                logger.info(f"  {m1:15} ⟷ {m2:15}: ρ = {rho:6.3f} (p = {pval:.3e})")

    # NaN handling summary
    logger.info("\n" + "=" * 80)
    logger.info("Missing Data (NaN) Handling Summary")
    logger.info("=" * 80)

    nan_counts = {
        "WASPAS": sum(1 for s in result_waspas.scores if np.isnan(s)),
        "TOPSIS": sum(1 for s in result_topsis.scores if np.isnan(s)),
        "PROMETHEE II": sum(1 for s in result_promethee2.scores if np.isnan(s)),
    }

    logger.info(f"\nProvincial NaN scores (all-NaN rows):")
    for method, count in nan_counts.items():
        logger.info(f"  {method:15}: {count} provinces")

    logger.info(f"\nTotal missing sub-criteria in decision matrix:")
    n_missing = np.isnan(ifs_matrix.mu).sum()
    total_cells = ifs_matrix.mu.size
    pct = 100.0 * n_missing / total_cells
    logger.info(f"  {n_missing} / {total_cells} cells ({pct:.2f}%)")
    logger.info(f"  (Automatically handled via weight re-normalisation)")

    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("Saving Results")
    logger.info("=" * 80)

    output_dir = Path(config.output.mcdm_dir) / "rankings"
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, result in [
        ("waspas", result_waspas),
        ("topsis", result_topsis),
        ("promethee2", result_promethee2),
    ]:
        filename = output_dir / f"{method}_rankings_{year}.csv"
        save_ranking_results(result, filename)
        logger.info(f"  {method:15}: {filename}")

    logger.info("\n" + "=" * 80)
    logger.info("Demo complete!")
    logger.info("=" * 80)


def _format_top_n(result, n: int) -> str:
    """Format top n provinces as string."""
    top_indices = [i for i, r in enumerate(result.ranks) if r <= n]
    top_indices.sort(key=lambda i: result.ranks[i])
    top = [f"{result.provinces[i]}({result.ranks[i]})" for i in top_indices]
    return ", ".join(top)


def _create_comparison_df(result_waspas, result_topsis, result_promethee2) -> pd.DataFrame:
    """Create comparison dataframe with rankings from all methods."""
    df = pd.DataFrame({
        "Province": result_waspas.provinces,
        "WASPAS_Rank": result_waspas.ranks,
        "WASPAS_Score": result_waspas.scores,
        "TOPSIS_Rank": result_topsis.ranks,
        "TOPSIS_Score": result_topsis.scores,
        "PROMETHEE_Rank": result_promethee2.ranks,
        "PROMETHEE_Score": result_promethee2.scores,
    })

    # Calculate average rank
    rank_cols = ["WASPAS_Rank", "TOPSIS_Rank", "PROMETHEE_Rank"]
    df["Avg_Rank"] = df[rank_cols].mean(axis=1)
    df["Rank_Std"] = df[rank_cols].std(axis=1)

    # Sort by average rank
    df = df.sort_values("Avg_Rank").reset_index(drop=True)

    return df


if __name__ == "__main__":
    main()
