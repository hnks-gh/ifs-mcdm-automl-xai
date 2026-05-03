"""
Phase 10: Visualization Demo & Showcase

Demonstrates complete visualization pipeline for MCDM and ML analysis results.
Generates all publication-quality figures and comprehensive CSV tables.

This script serves as both a demo and integration test for Phase 10 outputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_mock_mcdm_data():
    """Generate realistic mock MCDM analysis data."""
    logger.info("Generating mock MCDM data...")
    
    # Criteria weights (8 criteria × 14 years)
    criteria_weights = pd.DataFrame(
        np.random.dirichlet(np.ones(8), 14) * 0.3,
        index=range(2011, 2025),
        columns=[f'C{i:02d}' for i in range(1, 9)],
    )
    
    # Sub-criteria weights (29 subcriteria × 14 years)
    subcriteria_weights = pd.DataFrame(
        np.random.dirichlet(np.ones(29), 14) * 0.3,
        index=range(2011, 2025),
        columns=[f'SC{i:02d}' for i in range(11, 40)],
    )
    
    # Rankings for each method (63 provinces × 14 years)
    np.random.seed(42)
    rankings_dict = {}
    for method in ['if_waspas', 'if_topsis', 'if_promethee2']:
        # Create realistic rankings (prefer stability)
        base_ranks = np.arange(1, 64)
        rankings = []
        current = base_ranks.copy()
        
        for year in range(14):
            # Small perturbations to create temporal coherence
            permutation = np.random.permutation(63)
            indices = np.random.choice(63, min(5, 63), replace=False)
            current_year = current.copy()
            for i in indices:
                current_year[i] = current[permutation[i]]
            rankings.append(current_year)
            current = current_year
        
        rankings_dict[method] = pd.DataFrame(
            np.array(rankings).T,
            index=[f'P{i:02d}' for i in range(1, 64)],
            columns=range(2011, 2025)
        )
    
    # Scores (same provinces × years)
    scores_dict = {
        method: pd.DataFrame(
            np.random.uniform(0.2, 0.95, (63, 14)),
            index=[f'P{i:02d}' for i in range(1, 64)],
            columns=range(2011, 2025)
        )
        for method in rankings_dict.keys()
    }
    
    # Temporal stability results
    stability_results = {
        'rmsd': np.random.uniform(0.02, 0.08, 10),  # Reasonable stability
        'cv': {
            f'SC{i:02d}': {
                'mean': np.random.uniform(0.15, 0.25),
                'std': np.random.uniform(0.02, 0.05)
            }
            for i in range(11, 40)
        },
        'subcriteria': [f'SC{i:02d}' for i in range(11, 40)],
        'windows': [[2011+i, 2015+i] for i in range(10)],
    }
    
    # Sensitivity analysis results (Monte Carlo, 10k simulations)
    sensitivity_results = {
        'if_waspas': np.random.beta(8, 2, 10000),  # High concentration
        'if_topsis': np.random.beta(8, 2, 10000),
        'if_promethee2': np.random.beta(8, 2, 10000),
    }
    
    return {
        'criteria_weights': criteria_weights,
        'subcriteria_weights': subcriteria_weights,
        'rankings_dict': rankings_dict,
        'scores_dict': scores_dict,
        'stability_results': stability_results,
        'sensitivity_results': sensitivity_results,
    }


def generate_mock_ml_data():
    """Generate realistic mock ML analysis data."""
    logger.info("Generating mock ML data...")
    
    # Imputation statistics
    before_stats = pd.DataFrame({
        'Province': [f'P{i:02d}' for i in range(1, 15)],
        'Missing Cells': np.random.randint(0, 30, 14),
        'Missing Rows': np.random.randint(0, 3, 14),
        'Missing Percentage': np.random.uniform(0, 15, 14),
    }).set_index('Province')
    
    after_stats = pd.DataFrame(
        np.zeros_like(before_stats),
        index=before_stats.index,
        columns=before_stats.columns
    )
    
    # 2025 forecast
    forecast_2025 = pd.DataFrame(
        np.random.uniform(0.5, 3.2, (63, 29)),
        index=[f'P{i:02d}' for i in range(1, 64)],
        columns=[f'SC{i:02d}' for i in range(11, 40)],
    )
    
    # 2024 historical (for comparison)
    historical_2024 = pd.DataFrame(
        np.random.uniform(0.5, 3.2, (63, 29)),
        index=[f'P{i:02d}' for i in range(1, 64)],
        columns=[f'SC{i:02d}' for i in range(11, 40)],
    )
    
    # SHAP importance (28 features for each of 29 targets, aggregate)
    shap_importance = {
        f'SC{i:02d}': np.random.exponential(0.1)
        for i in range(11, 40)
    }
    
    # Normalize importance scores
    total = sum(shap_importance.values())
    shap_importance = {k: v/total for k, v in shap_importance.items()}
    
    # Optional: SHAP values and forecast for waterfall plots
    shap_values_dict = {
        f'SC{i:02d}': np.random.normal(0, 0.05, 63)
        for i in range(11, 40)
    }
    
    return {
        'before_imputation_stats': before_stats,
        'after_imputation_stats': after_stats,
        'forecast_2025': forecast_2025,
        'historical_2024': historical_2024,
        'shap_importance': shap_importance,
        'shap_values_dict': shap_values_dict,
        'forecast_with_shap': forecast_2025,
        'shap_with_values': pd.DataFrame(shap_values_dict, index=forecast_2025.index),
        'base_value': 1.65,  # Mid-range score
    }


def main():
    """Run Phase 10 visualization demo."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 10: VISUALIZATION DEMO & SHOWCASE")
    logger.info("=" * 80)
    
    # Setup output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Generate mock data
        logger.info("\n" + "-" * 80)
        logger.info("STEP 1: GENERATING MOCK DATA")
        logger.info("-" * 80)
        
        mcdm_data = generate_mock_mcdm_data()
        logger.info("✓ MCDM data generated")
        logger.info(f"  - Criteria weights: {mcdm_data['criteria_weights'].shape}")
        logger.info(f"  - Sub-criteria weights: {mcdm_data['subcriteria_weights'].shape}")
        logger.info(f"  - Ranking methods: {list(mcdm_data['rankings_dict'].keys())}")
        
        ml_data = generate_mock_ml_data()
        logger.info("✓ ML data generated")
        logger.info(f"  - Forecast 2025: {ml_data['forecast_2025'].shape}")
        logger.info(f"  - SHAP importance features: {len(ml_data['shap_importance'])}")
        
        # Generate visualizations
        logger.info("\n" + "-" * 80)
        logger.info("STEP 2: GENERATING VISUALIZATIONS")
        logger.info("-" * 80)
        
        from src.pipeline.visualization_orchestration import (
            orchestrate_phase10_visualizations
        )
        
        summary = orchestrate_phase10_visualizations(
            output_base_dir=output_dir,
            mcdm_data={
                'criteria_weights': mcdm_data['criteria_weights'],
                'subcriteria_weights': mcdm_data['subcriteria_weights'],
                'rankings_dict': mcdm_data['rankings_dict'],
                'scores_dict': mcdm_data['scores_dict'],
                'stability_results': mcdm_data['stability_results'],
                'sensitivity_results': mcdm_data['sensitivity_results'],
            },
            ml_data={
                'before_imputation_stats': ml_data['before_imputation_stats'],
                'after_imputation_stats': ml_data['after_imputation_stats'],
                'forecast_2025': ml_data['forecast_2025'],
                'historical_2024': ml_data['historical_2024'],
                'shap_importance': ml_data['shap_importance'],
                'shap_values_dict': ml_data['shap_values_dict'],
                'forecast_with_shap': ml_data['forecast_with_shap'],
                'shap_with_values': ml_data['shap_with_values'],
                'base_value': ml_data['base_value'],
            }
        )
        
        # Display summary
        logger.info("\n" + "-" * 80)
        logger.info("STEP 3: SUMMARY OF GENERATED OUTPUTS")
        logger.info("-" * 80)
        logger.info(f"\nTotal figures generated:    {summary.total_figures}")
        logger.info(f"Total tables generated:     {summary.total_tables}")
        logger.info(f"\nOutput directories:")
        logger.info(f"  Figures: {summary.output_directories['figures_base']}")
        logger.info(f"  Tables:  {summary.output_directories['tables_base']}")
        
        logger.info("\n" + "-" * 80)
        logger.info("FIGURES BY CATEGORY")
        logger.info("-" * 80)
        
        weighting_figs = [k for k in summary.all_figure_paths.keys() if 'weighting' in k or 'weight' in k]
        ranking_figs = [k for k in summary.all_figure_paths.keys() if any(x in k for x in ['ranking', 'rank', 'spearman', 'iqr', 'yoy', 'bump'])]
        ml_figs = [k for k in summary.all_figure_paths.keys() if any(x in k for x in ['ml', 'imputation', 'forecast', 'shap'])]
        
        logger.info(f"\nWeighting Analysis ({len(weighting_figs)} figures):")
        for fig in weighting_figs:
            logger.info(f"  ✓ {fig}")
        
        logger.info(f"\nRanking Analysis ({len(ranking_figs)} figures):")
        for fig in ranking_figs:
            logger.info(f"  ✓ {fig}")
        
        logger.info(f"\nML Analysis ({len(ml_figs)} figures):")
        for fig in ml_figs:
            logger.info(f"  ✓ {fig}")
        
        logger.info("\n" + "-" * 80)
        logger.info("TABLES BY CATEGORY")
        logger.info("-" * 80)
        
        weighting_tables = [k for k in summary.all_table_paths.keys() if 'weighting' in k or 'weight' in k]
        ranking_tables = [k for k in summary.all_table_paths.keys() if any(x in k for x in ['ranking', 'rank', 'spearman', 'iqr', 'yoy'])]
        ml_tables = [k for k in summary.all_table_paths.keys() if any(x in k for x in ['ml', 'imputation', 'forecast', 'shap'])]
        
        logger.info(f"\nWeighting Analysis ({len(weighting_tables)} tables):")
        for table in weighting_tables:
            logger.info(f"  ✓ {table}")
        
        logger.info(f"\nRanking Analysis ({len(ranking_tables)} tables):")
        for table in ranking_tables:
            logger.info(f"  ✓ {table}")
        
        logger.info(f"\nML Analysis ({len(ml_tables)} tables):")
        for table in ml_tables:
            logger.info(f"  ✓ {table}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ PHASE 10 VISUALIZATION DEMO COMPLETED SUCCESSFULLY")
        logger.info("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Error during visualization generation: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
