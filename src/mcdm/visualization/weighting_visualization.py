"""
Weighting Analysis Visualization & Export Module

Generates publication-quality figures and comprehensive data tables for:
1. Criteria and sub-criteria weight heatmaps
2. Temporal weight evolution profiles
3. Temporal stability metrics (RMSD, CV)
4. Sensitivity analysis results (Kendall tau-b distributions)

Outputs:
- Figures: PNG at 300 DPI (6 visualizations)
- Tables: CSV format with full precision and structured layout
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass

from ...utils.plot_utils import (
    PlotConfig, ColorPalette, set_publication_style,
    save_figure, save_dataframe_csv,
    plot_heatmap, plot_multiline, plot_boxplot, plot_radar_grid
)

logger = logging.getLogger(__name__)


@dataclass
class WeightingVisualizationResult:
    """Container for weighting visualization outputs."""
    figures: Dict[str, Path]
    tables: Dict[str, Path]
    summary_metrics: Dict[str, float]


class WeightingVisualizer:
    """Generates all weighting analysis visualizations and exports."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Base output directory (e.g., 'output/')
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures" / "weighting"
        self.tables_dir = self.output_dir / "tables" / "weighting"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        set_publication_style()
    
    # ========================================================================
    # 1. WEIGHT HEATMAPS (Criteria & Sub-criteria)
    # ========================================================================
    
    def visualize_criteria_weights(self, criteria_weights: pd.DataFrame) -> Tuple[Path, Path]:
        """
        Create heatmap of criteria weights over time + CSV export.
        
        Args:
            criteria_weights: DataFrame shape (n_years, 8) with criteria weights
            
        Returns:
            Tuple of (figure_path, table_path)
        """
        # Ensure numeric columns
        criteria_weights = criteria_weights.apply(pd.to_numeric, errors='coerce')
        
        # Save CSV table
        table_path = save_dataframe_csv(
            criteria_weights,
            self.tables_dir / "w01_criteria_weights.csv",
            description="Criteria (C01-C08) weights by year"
        )
        
        # Create heatmap
        fig = plot_heatmap(
            criteria_weights,
            title="IF-CRITIC Criteria Weights (C01–C08) | 2011–2024",
            xlabel="Criteria",
            ylabel="Year",
            figsize=PlotConfig.FIGURE_MEDIUM,
            cmap="RdYlGn",
            cbar_label="Weight",
            annot=True,
            fmt='.4f',
            vmin=0.0,
            vmax=0.3
        )
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "w01_criteria_weights_heatmap.png",
            title="Criteria Weights Heatmap"
        )
        plt.close(fig)
        
        return figure_path, table_path
    
    def visualize_subcriteria_weights(self, subcriteria_weights: pd.DataFrame) -> Tuple[Path, Path]:
        """
        Create heatmap of sub-criteria weights over time + CSV export.
        
        Args:
            subcriteria_weights: DataFrame shape (n_years, 29) with subcriteria weights
            
        Returns:
            Tuple of (figure_path, table_path)
        """
        # Ensure numeric columns
        subcriteria_weights = subcriteria_weights.apply(pd.to_numeric, errors='coerce')
        
        # Save CSV table
        table_path = save_dataframe_csv(
            subcriteria_weights,
            self.tables_dir / "w02_subcriteria_weights.csv",
            description="Sub-criteria (SC11-SC83) weights by year"
        )
        
        # For large heatmap, use larger figure
        fig = plot_heatmap(
            subcriteria_weights,
            title="IF-CRITIC Sub-Criteria Weights (SC11–SC83) | 2011–2024",
            xlabel="Sub-Criteria",
            ylabel="Year",
            figsize=(16, 10),
            cmap="viridis",
            cbar_label="Weight",
            annot=False,  # Too dense for annotations
            vmin=0.0,
            vmax=0.15
        )
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "w02_subcriteria_weights_heatmap.png",
            title="Sub-Criteria Weights Heatmap"
        )
        plt.close(fig)
        
        return figure_path, table_path
    
    # ========================================================================
    # 2. RADAR GRID (Annual Weight Profiles)
    # ========================================================================
    
    def visualize_radar_annual(self, criteria_weights: pd.DataFrame) -> Path:
        """
        Create 14-panel radar chart (one per year) showing criteria weight profiles.
        
        Args:
            criteria_weights: DataFrame shape (n_years, 8)
            
        Returns:
            Figure path
        """
        # Prepare data for radar plots
        radar_data = {}
        for year in criteria_weights.index:
            radar_data[f"Year {year}"] = dict(zip(
                criteria_weights.columns,
                criteria_weights.loc[year].values
            ))
        
        # Create grid
        fig = plot_radar_grid(
            radar_data,
            title="IF-CRITIC Criteria Weight Profiles by Year (2011–2024)",
            figsize=(16, 14),
            ncols=4
        )
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "w03_weights_radar_grid.png",
            title="Annual Radar Weight Profiles"
        )
        plt.close(fig)
        
        return figure_path
    
    # ========================================================================
    # 3. TEMPORAL TRENDS (Line plots)
    # ========================================================================
    
    def visualize_temporal_trends(self, subcriteria_weights: pd.DataFrame) -> Tuple[Path, Path]:
        """
        Create line plots showing weight evolution + CSV export.
        
        Args:
            subcriteria_weights: DataFrame shape (n_years, 29)
            
        Returns:
            Tuple of (figure_path, table_path)
        """
        # CSV already saved in visualize_subcriteria_weights, reuse
        # Here we create trend visualization
        
        # Select 8 key sub-criteria (first of each criterion)
        key_cols = [f"SC{i}1" for i in range(1, 9)]
        key_weights = subcriteria_weights[[c for c in key_cols if c in subcriteria_weights.columns]]
        
        # Create line plot
        fig = plot_multiline(
            key_weights,
            title="Temporal Evolution of Key Sub-Criteria Weights (2011–2024)",
            xlabel="Year",
            ylabel="Weight",
            figsize=PlotConfig.FIGURE_MEDIUM,
            include_legend=True
        )
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "w04_weights_temporal_trends.png",
            title="Weight Trends Over Time"
        )
        plt.close(fig)
        
        return figure_path, None  # None because CSV already saved
    
    # ========================================================================
    # 4. TEMPORAL STABILITY (RMSD & CV)
    # ========================================================================
    
    def visualize_temporal_stability(self, stability_results: Dict[str, Any]) -> Tuple[Path, Path]:
        """
        Create visualization of temporal stability metrics + CSV export.
        
        Args:
            stability_results: Dict with 'rmsd', 'cv', 'windows', 'subcriteria' keys
            
        Returns:
            Tuple of (figure_path, table_path)
        """
        # Extract data
        rmsd_values = stability_results.get('rmsd', [])
        cv_values = stability_results.get('cv', {})
        subcriteria = stability_results.get('subcriteria', [])
        windows = stability_results.get('windows', [])
        
        # Prepare DataFrame for CSV
        stability_df = pd.DataFrame({
            'Sub-Criteria': list(cv_values.keys()),
            'Mean CV': [v['mean'] for v in cv_values.values()],
            'Std CV': [v['std'] for v in cv_values.values()],
            'Mean RMSD': [np.mean(rmsd_values)] * len(cv_values),
        })
        
        # Save CSV
        table_path = save_dataframe_csv(
            stability_df,
            self.tables_dir / "w03_temporal_stability.csv",
            description="Temporal stability metrics (RMSD, CV)"
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=PlotConfig.FIGURE_WIDE, facecolor='white')
        
        # RMSD line plot
        ax = axes[0]
        ax.plot(range(len(rmsd_values)), rmsd_values, 'o-', linewidth=2.5, markersize=8, color='#1f77b4')
        ax.fill_between(range(len(rmsd_values)), rmsd_values, alpha=0.3, color='#1f77b4')
        ax.set_title("RMSD Between Consecutive Windows", fontsize=12, fontweight='bold')
        ax.set_xlabel("Window Transition", fontsize=11)
        ax.set_ylabel("RMSD", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # CV bar plot
        ax = axes[1]
        cv_means = [cv_values[k]['mean'] for k in sorted(cv_values.keys())]
        cv_stds = [cv_values[k]['std'] for k in sorted(cv_values.keys())]
        
        ax.bar(range(len(cv_means)), cv_means, yerr=cv_stds, capsize=5, 
               color='#ff7f0e', edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.set_title("Coefficient of Variation (CV) by Sub-Criteria", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sub-Criteria", fontsize=11)
        ax.set_ylabel("CV", fontsize=11)
        ax.set_xticks(range(len(cv_means)))
        ax.set_xticklabels([f"SC{i+11}" for i in range(len(cv_means))], rotation=45)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        fig.suptitle("Temporal Stability Analysis (10 Overlapping Windows)", 
                     fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', y=0.98)
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "w05_temporal_stability_rmsd.png",
            title="Temporal Stability Metrics"
        )
        plt.close(fig)
        
        return figure_path, table_path
    
    # ========================================================================
    # 5. SENSITIVITY ANALYSIS (Monte Carlo)
    # ========================================================================
    
    def visualize_sensitivity_analysis(self, sensitivity_results: Dict[str, Any]) -> Tuple[Path, Path]:
        """
        Create boxplot of Kendall tau-b distributions + CSV export.
        
        Args:
            sensitivity_results: Dict with method -> tau_b_values mapping
            
        Returns:
            Tuple of (figure_path, table_path)
        """
        # Prepare data for table
        summary_stats = {}
        for method, tau_b_values in sensitivity_results.items():
            summary_stats[method] = {
                'Mean τ-b': np.mean(tau_b_values),
                'Std τ-b': np.std(tau_b_values),
                'Min τ-b': np.min(tau_b_values),
                '5th Percentile': np.percentile(tau_b_values, 5),
                'Median τ-b': np.median(tau_b_values),
                '95th Percentile': np.percentile(tau_b_values, 95),
                'Max τ-b': np.max(tau_b_values),
            }
        
        summary_df = pd.DataFrame(summary_stats).T
        
        # Save CSV
        table_path = save_dataframe_csv(
            summary_df,
            self.tables_dir / "w04_sensitivity_results.csv",
            description="Monte Carlo sensitivity analysis (Kendall tau-b)"
        )
        
        # Create boxplot
        fig = plot_boxplot(
            sensitivity_results,
            title="Sensitivity Analysis: Ranking Stability under Weight Perturbation\n"
                  "(10,000 Monte Carlo simulations | Dirichlet perturbation)",
            ylabel="Kendall's τ-b (weighted)",
            figsize=PlotConfig.FIGURE_MEDIUM,
            color_map=PlotConfig.METHOD_COLORS
        )
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "w06_sensitivity_analysis_boxplot.png",
            title="Sensitivity Analysis Distribution"
        )
        plt.close(fig)
        
        return figure_path, table_path
    
    # ========================================================================
    # ORCHESTRATION
    # ========================================================================
    
    def generate_all_visualizations(self,
                                   criteria_weights: pd.DataFrame,
                                   subcriteria_weights: pd.DataFrame,
                                   stability_results: Dict[str, Any],
                                   sensitivity_results: Dict[str, Any]) -> WeightingVisualizationResult:
        """
        Generate all weighting visualizations and exports.
        
        Args:
            criteria_weights: Criteria weights by year
            subcriteria_weights: Sub-criteria weights by year
            stability_results: Temporal stability metrics
            sensitivity_results: Monte Carlo sensitivity results
            
        Returns:
            WeightingVisualizationResult with all outputs
        """
        logger.info("=" * 70)
        logger.info("WEIGHTING ANALYSIS: GENERATING VISUALIZATIONS & EXPORTS")
        logger.info("=" * 70)
        
        figures = {}
        tables = {}
        
        # 1. Criteria weights
        fig_path, table_path = self.visualize_criteria_weights(criteria_weights)
        figures['criteria_heatmap'] = fig_path
        tables['criteria_weights'] = table_path
        
        # 2. Sub-criteria weights
        fig_path, table_path = self.visualize_subcriteria_weights(subcriteria_weights)
        figures['subcriteria_heatmap'] = fig_path
        tables['subcriteria_weights'] = table_path
        
        # 3. Radar grid
        fig_path = self.visualize_radar_annual(criteria_weights)
        figures['radar_grid'] = fig_path
        
        # 4. Temporal trends
        fig_path, _ = self.visualize_temporal_trends(subcriteria_weights)
        figures['temporal_trends'] = fig_path
        
        # 5. Temporal stability
        fig_path, table_path = self.visualize_temporal_stability(stability_results)
        figures['stability_metrics'] = fig_path
        tables['stability'] = table_path
        
        # 6. Sensitivity analysis
        fig_path, table_path = self.visualize_sensitivity_analysis(sensitivity_results)
        figures['sensitivity_boxplot'] = fig_path
        tables['sensitivity'] = table_path
        
        logger.info(f"\n✓ Generated {len(figures)} figures")
        logger.info(f"✓ Generated {len(tables)} tables")
        
        return WeightingVisualizationResult(
            figures=figures,
            tables=tables,
            summary_metrics={
                'n_figures': len(figures),
                'n_tables': len(tables),
            }
        )
