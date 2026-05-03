"""
Phase 10: Visualization Orchestration & Output Management

Coordinates all visualization generation across MCDM and ML pipelines.
Produces comprehensive outputs:
- Elegant publication-quality PNG figures (300 DPI)
- Detailed CSV tables with full precision
- Validation and integrity reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass
from datetime import datetime

from ..mcdm.visualization import WeightingVisualizer, RankingVisualizer
from ..ml.visualization import MLVisualizer

logger = logging.getLogger(__name__)


@dataclass
class VisualizationSummary:
    """Summary of all visualization outputs."""
    timestamp: str
    total_figures: int
    total_tables: int
    mcdm_figures: int
    mcdm_tables: int
    ml_figures: int
    ml_tables: int
    output_directories: Dict[str, Path]
    all_figure_paths: Dict[str, Path]
    all_table_paths: Dict[str, Path]


class VisualizationOrchestrator:
    """Orchestrates all Phase 10 visualization generation."""
    
    def __init__(self, output_base_dir: Path = Path("output")):
        """
        Initialize orchestrator.
        
        Args:
            output_base_dir: Base output directory
        """
        self.output_dir = Path(output_base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize sub-visualizers
        self.weighting_viz = WeightingVisualizer(self.output_dir)
        self.ranking_viz = RankingVisualizer(self.output_dir)
        self.ml_viz = MLVisualizer(self.output_dir)
        
        logger.info(f"VisualizationOrchestrator initialized at {self.output_dir}")
    
    # ========================================================================
    # MCDM VISUALIZATION ORCHESTRATION
    # ========================================================================
    
    def generate_mcdm_visualizations(self,
                                    criteria_weights: pd.DataFrame,
                                    subcriteria_weights: pd.DataFrame,
                                    rankings_dict: Dict[str, pd.DataFrame],
                                    scores_dict: Dict[str, pd.DataFrame],
                                    stability_results: Dict[str, Any],
                                    sensitivity_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate all MCDM visualizations.
        
        Args:
            criteria_weights: Criteria weights by year
            subcriteria_weights: Sub-criteria weights by year
            rankings_dict: Dict[method] -> rankings DataFrame
            scores_dict: Dict[method] -> scores DataFrame
            stability_results: Temporal stability metrics
            sensitivity_results: Sensitivity analysis results
            
        Returns:
            Dict with MCDM visualization results
        """
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 10: MCDM VISUALIZATION")
        logger.info("=" * 70)
        
        # Weighting visualizations
        weighting_result = self.weighting_viz.generate_all_visualizations(
            criteria_weights,
            subcriteria_weights,
            stability_results,
            sensitivity_results
        )
        
        # Ranking visualizations
        ranking_result = self.ranking_viz.generate_all_visualizations(
            rankings_dict,
            scores_dict
        )
        
        return {
            'weighting': weighting_result,
            'ranking': ranking_result,
        }
    
    # ========================================================================
    # ML VISUALIZATION ORCHESTRATION
    # ========================================================================
    
    def generate_ml_visualizations(self,
                                  before_imputation_stats: pd.DataFrame,
                                  after_imputation_stats: pd.DataFrame,
                                  forecast_2025: pd.DataFrame,
                                  historical_2024: pd.DataFrame,
                                  shap_importance: Dict[str, float],
                                  shap_values_dict: Optional[Dict[str, np.ndarray]] = None,
                                  forecast_with_shap: Optional[pd.DataFrame] = None,
                                  shap_with_values: Optional[pd.DataFrame] = None,
                                  base_value: float = 0.0) -> Dict[str, Any]:
        """
        Generate all ML visualizations.
        
        Args:
            before_imputation_stats: Pre-imputation statistics
            after_imputation_stats: Post-imputation statistics
            forecast_2025: 2025 forecast DataFrame
            historical_2024: 2024 historical data
            shap_importance: SHAP importance scores
            shap_values_dict: Optional SHAP value arrays
            forecast_with_shap: Optional forecast for waterfall
            shap_with_values: Optional SHAP values for waterfall
            base_value: Base value for waterfall plots
            
        Returns:
            Dict with ML visualization results
        """
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 10: ML VISUALIZATION")
        logger.info("=" * 70)
        
        ml_result = self.ml_viz.generate_all_visualizations(
            before_imputation_stats,
            after_imputation_stats,
            forecast_2025,
            historical_2024,
            shap_importance,
            shap_values_dict,
            forecast_with_shap,
            shap_with_values,
            base_value
        )
        
        return {'ml': ml_result}
    
    # ========================================================================
    # COMPREHENSIVE SUMMARY & VALIDATION
    # ========================================================================
    
    def generate_visualization_manifest(self,
                                       mcdm_results: Dict[str, Any],
                                       ml_results: Dict[str, Any]) -> VisualizationSummary:
        """
        Create comprehensive manifest of all visualizations.
        
        Args:
            mcdm_results: MCDM visualization results
            ml_results: ML visualization results
            
        Returns:
            VisualizationSummary with all metadata
        """
        # Aggregate all paths
        all_figures = {}
        all_tables = {}
        
        # MCDM
        if 'weighting' in mcdm_results:
            weighting_viz = mcdm_results['weighting']
            all_figures.update(weighting_viz.figures)
            all_tables.update(weighting_viz.tables)
        
        if 'ranking' in mcdm_results:
            ranking_viz = mcdm_results['ranking']
            all_figures.update(ranking_viz.figures)
            all_tables.update(ranking_viz.tables)
        
        # ML
        if 'ml' in ml_results:
            ml_viz = ml_results['ml']
            all_figures.update(ml_viz.figures)
            all_tables.update(ml_viz.tables)
        
        # Create directories metadata
        output_dirs = {
            'figures_base': self.output_dir / "figures",
            'tables_base': self.output_dir / "tables",
            'weighting_figures': self.output_dir / "figures" / "weighting",
            'ranking_figures': self.output_dir / "figures" / "ranking",
            'ml_figures': self.output_dir / "figures" / "ml",
            'weighting_tables': self.output_dir / "tables" / "weighting",
            'ranking_tables': self.output_dir / "tables" / "ranking",
            'ml_tables': self.output_dir / "tables" / "ml",
        }
        
        summary = VisualizationSummary(
            timestamp=self.timestamp,
            total_figures=len(all_figures),
            total_tables=len(all_tables),
            mcdm_figures=len([f for f in all_figures if any(k in str(f) for k in ['weighting', 'ranking'])]),
            mcdm_tables=len([t for t in all_tables if any(k in str(t) for k in ['weighting', 'ranking'])]),
            ml_figures=len([f for f in all_figures if 'ml' in str(f)]),
            ml_tables=len([t for t in all_tables if 'ml' in str(t)]),
            output_directories=output_dirs,
            all_figure_paths=all_figures,
            all_table_paths=all_tables,
        )
        
        return summary
    
    def create_manifest_csv(self, summary: VisualizationSummary) -> Path:
        """
        Create CSV manifest of all outputs.
        
        Args:
            summary: VisualizationSummary
            
        Returns:
            Path to manifest file
        """
        manifest_dir = self.output_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        
        # Figures manifest
        figures_data = []
        for name, path in sorted(summary.all_figure_paths.items()):
            figures_data.append({
                'Output': name,
                'Type': 'Figure (PNG)',
                'Path': str(path),
                'Category': self._categorize_output(name),
            })
        
        figures_df = pd.DataFrame(figures_data)
        figures_path = manifest_dir / f"manifest_figures_{self.timestamp}.csv"
        figures_df.to_csv(figures_path, index=False)
        
        # Tables manifest
        tables_data = []
        for name, path in sorted(summary.all_table_paths.items()):
            tables_data.append({
                'Output': name,
                'Type': 'Table (CSV)',
                'Path': str(path),
                'Category': self._categorize_output(name),
            })
        
        tables_df = pd.DataFrame(tables_data)
        tables_path = manifest_dir / f"manifest_tables_{self.timestamp}.csv"
        tables_df.to_csv(tables_path, index=False)
        
        # Summary manifest
        summary_data = {
            'Metric': [
                'Timestamp',
                'Total Figures',
                'Total Tables',
                'MCDM Figures',
                'MCDM Tables',
                'ML Figures',
                'ML Tables',
                'Weighting Visualizations',
                'Ranking Visualizations',
                'ML Visualizations',
            ],
            'Value': [
                summary.timestamp,
                summary.total_figures,
                summary.total_tables,
                summary.mcdm_figures,
                summary.mcdm_tables,
                summary.ml_figures,
                summary.ml_tables,
                '6 figures + 6 tables',
                '7 figures + 4 tables',
                '5+ figures + 4 tables',
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = manifest_dir / f"manifest_summary_{self.timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"\n✓ Manifest files created in {manifest_dir}")
        
        return summary_path
    
    def _categorize_output(self, name: str) -> str:
        """Categorize output name."""
        if 'weighting' in name or 'weight' in name:
            return 'Weighting Analysis'
        elif 'ranking' in name or 'rank' in name or 'spearman' in name or 'iqr' in name or 'yoy' in name or 'bump' in name:
            return 'Ranking Analysis'
        elif 'ml' in name or 'imputation' in name or 'forecast' in name or 'shap' in name:
            return 'ML Analysis'
        else:
            return 'Other'
    
    def validate_outputs(self, summary: VisualizationSummary) -> Dict[str, bool]:
        """
        Validate all generated outputs.
        
        Args:
            summary: VisualizationSummary
            
        Returns:
            Dict with validation results
        """
        validation = {
            'all_figures_exist': True,
            'all_tables_exist': True,
            'directories_exist': True,
            'png_quality': True,
        }
        
        # Check figures
        for name, path in summary.all_figure_paths.items():
            if not Path(path).exists():
                logger.warning(f"✗ Figure missing: {name}")
                validation['all_figures_exist'] = False
            elif not str(path).endswith('.png'):
                validation['png_quality'] = False
        
        # Check tables
        for name, path in summary.all_table_paths.items():
            if not Path(path).exists():
                logger.warning(f"✗ Table missing: {name}")
                validation['all_tables_exist'] = False
        
        # Check directories
        for dir_name, dir_path in summary.output_directories.items():
            if not dir_path.exists():
                logger.warning(f"✗ Directory missing: {dir_name}")
                validation['directories_exist'] = False
        
        return validation
    
    def generate_validation_report(self, summary: VisualizationSummary,
                                  validation: Dict[str, bool]) -> Path:
        """
        Generate validation report.
        
        Args:
            summary: VisualizationSummary
            validation: Validation results
            
        Returns:
            Path to validation report
        """
        reports_dir = self.output_dir / "validation"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'Check': list(validation.keys()),
            'Status': ['✓ PASS' if v else '✗ FAIL' for v in validation.values()],
        }
        
        report_df = pd.DataFrame(report_data)
        report_path = reports_dir / f"phase10_validation_{self.timestamp}.csv"
        report_df.to_csv(report_path, index=False)
        
        logger.info(f"\n✓ Validation report: {report_path}")
        
        return report_path
    
    def print_summary(self, summary: VisualizationSummary, validation: Dict[str, bool]):
        """Print comprehensive summary."""
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 10: VISUALIZATION GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nTimestamp: {summary.timestamp}")
        logger.info(f"\nOutput Summary:")
        logger.info(f"  Total Figures:    {summary.total_figures}")
        logger.info(f"  Total Tables:     {summary.total_tables}")
        logger.info(f"\nMCDM Outputs:")
        logger.info(f"  Weighting:        6 figures + 6 tables")
        logger.info(f"  Ranking:          7 figures + 4 tables")
        logger.info(f"\nML Outputs:")
        logger.info(f"  Forecasting:      2 figures + 2 tables")
        logger.info(f"  SHAP:             3 figures + 1 table")
        logger.info(f"\nOutput Directories:")
        logger.info(f"  Figures:  {summary.output_directories['figures_base']}")
        logger.info(f"  Tables:   {summary.output_directories['tables_base']}")
        logger.info(f"\nValidation:")
        for check, result in validation.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"  {check.replace('_', ' ').title()}: {status}")
        logger.info("\n" + "=" * 70)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def orchestrate_phase10_visualizations(
    output_base_dir: Path = Path("output"),
    mcdm_data: Optional[Dict[str, Any]] = None,
    ml_data: Optional[Dict[str, Any]] = None,
) -> VisualizationSummary:
    """
    Convenience function to orchestrate all Phase 10 visualizations.
    
    Args:
        output_base_dir: Base output directory
        mcdm_data: Dict with MCDM analysis results
        ml_data: Dict with ML analysis results
        
    Returns:
        VisualizationSummary
    """
    orchestrator = VisualizationOrchestrator(output_base_dir)
    
    # Generate visualizations
    if mcdm_data:
        mcdm_results = orchestrator.generate_mcdm_visualizations(**mcdm_data)
    else:
        mcdm_results = {}
    
    if ml_data:
        ml_results = orchestrator.generate_ml_visualizations(**ml_data)
    else:
        ml_results = {}
    
    # Create summary
    summary = orchestrator.generate_visualization_manifest(mcdm_results, ml_results)
    
    # Validation
    validation = orchestrator.validate_outputs(summary)
    orchestrator.generate_validation_report(summary, validation)
    
    # Print summary
    orchestrator.print_summary(summary, validation)
    
    return summary
