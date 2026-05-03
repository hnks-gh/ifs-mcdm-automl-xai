"""
Ranking Analysis Visualization & Export Module

Generates publication-quality figures and comprehensive data tables for:
1. Ranking heatmaps (per method per year)
2. Inter-method Spearman correlation matrix
3. IQR-based discriminatory power metrics
4. Year-to-year temporal persistence of rankings
5. Top-10 provinces bump chart visualization

Outputs:
- Figures: PNG at 300 DPI (7 visualizations)
- Tables: CSV format with full precision
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from scipy.stats import spearmanr

from ...utils.plot_utils import (
    PlotConfig, ColorPalette, set_publication_style,
    save_figure, save_dataframe_csv,
    plot_heatmap, plot_correlation_heatmap, plot_grouped_bars, plot_multiline
)

logger = logging.getLogger(__name__)


@dataclass
class RankingVisualizationResult:
    """Container for ranking visualization outputs."""
    figures: Dict[str, Path]
    tables: Dict[str, Path]
    summary_metrics: Dict[str, float]


class RankingVisualizer:
    """Generates all ranking analysis visualizations and exports."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Base output directory (e.g., 'output/')
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures" / "ranking"
        self.tables_dir = self.output_dir / "tables" / "ranking"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        set_publication_style()
        self.method_colors = PlotConfig.METHOD_COLORS
    
    # ========================================================================
    # 1. RANKING HEATMAPS (Per method)
    # ========================================================================
    
    def visualize_ranking_heatmaps(self, rankings_dict: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Path], Path]:
        """
        Create heatmaps of province rankings per method + combined CSV export.
        
        Args:
            rankings_dict: Dict[method_name] -> DataFrame(provinces × years)
            
        Returns:
            Tuple of (method_figure_paths_dict, combined_table_path)
        """
        figure_paths = {}
        
        # Prepare combined ranking table
        combined_rankings = pd.DataFrame()
        
        for method, rankings_df in rankings_dict.items():
            # Save individual heatmap
            fig = plot_heatmap(
                rankings_df,
                title=f"{method.upper()} Province Rankings | 2011–2024",
                xlabel="Year",
                ylabel="Province",
                figsize=(14, 16),
                cmap="RdYlGn_r",
                cbar_label="Rank (1=Best)",
                annot=False,  # Too large for annotations
                vmin=1,
                vmax=63
            )
            
            figure_key = f"{method}_heatmap"
            figure_path = save_figure(
                fig,
                self.figures_dir / f"r0{list(rankings_dict.keys()).index(method) + 1}_{method}_ranking_heatmap.png",
                title=f"{method} Ranking Heatmap"
            )
            figure_paths[figure_key] = figure_path
            plt.close(fig)
            
            # Add to combined table (mean rank per province)
            combined_rankings[method] = rankings_df.mean(axis=1)
        
        # Save combined ranking table
        table_path = save_dataframe_csv(
            combined_rankings,
            self.tables_dir / "r01_all_rankings.csv",
            description="Average province rankings by method"
        )
        
        return figure_paths, table_path
    
    # ========================================================================
    # 2. INTER-METHOD CORRELATION MATRIX
    # ========================================================================
    
    def visualize_inter_method_correlation(self, rankings_dict: Dict[str, pd.DataFrame]) -> Tuple[Path, Path]:
        """
        Create Spearman correlation matrix between ranking methods + CSV export.
        
        Args:
            rankings_dict: Dict[method_name] -> DataFrame(provinces × years)
            
        Returns:
            Tuple of (figure_path, table_path)
        """
        # Flatten rankings to single vectors per method
        method_vectors = {}
        for method, rankings_df in rankings_dict.items():
            # Flatten across all years and provinces
            method_vectors[method] = rankings_df.values.flatten()
        
        # Compute pairwise Spearman correlations
        methods = list(method_vectors.keys())
        n_methods = len(methods)
        corr_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    rho, _ = spearmanr(method_vectors[method1], method_vectors[method2])
                    corr_matrix[i, j] = rho
        
        # Create DataFrame for heatmap and CSV
        corr_df = pd.DataFrame(
            corr_matrix,
            index=[m.upper() for m in methods],
            columns=[m.upper() for m in methods]
        )
        
        # Save CSV
        table_path = save_dataframe_csv(
            corr_df,
            self.tables_dir / "r02_inter_method_correlation.csv",
            description="Spearman correlation between ranking methods"
        )
        
        # Create heatmap
        fig = plot_correlation_heatmap(
            corr_df,
            title="Inter-Method Agreement: Spearman Correlation Matrix\n(IF-WASPAS vs IF-TOPSIS vs IF-PROMETHEE II)",
            figsize=PlotConfig.FIGURE_SQUARE
        )
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "r04_inter_method_correlation.png",
            title="Inter-Method Correlation Matrix"
        )
        plt.close(fig)
        
        return figure_path, table_path
    
    # ========================================================================
    # 3. DISCRIMINATORY POWER (IQR)
    # ========================================================================
    
    def visualize_discriminatory_power(self, scores_dict: Dict[str, pd.DataFrame]) -> Tuple[Path, Path]:
        """
        Create IQR-based discriminatory power visualization + CSV export.
        
        Args:
            scores_dict: Dict[method_name] -> DataFrame of scores (provinces × years)
            
        Returns:
            Tuple of (figure_path, table_path)
        """
        # Compute IQR per method per year
        iqr_data = {}
        for method, scores_df in scores_dict.items():
            iqr_per_year = scores_df.apply(lambda col: np.percentile(col, 75) - np.percentile(col, 25))
            iqr_data[method] = iqr_per_year
        
        iqr_df = pd.DataFrame(iqr_data)
        
        # Save CSV
        table_path = save_dataframe_csv(
            iqr_df,
            self.tables_dir / "r03_iqr_metrics.csv",
            description="IQR-based discriminatory power by method and year"
        )
        
        # Create grouped bar chart
        fig = plot_grouped_bars(
            iqr_df,
            title="Discriminatory Power of Ranking Methods (IQR of Scores)\nHigher IQR = Better Discrimination",
            xlabel="Year",
            ylabel="Interquartile Range (IQR)",
            figsize=PlotConfig.FIGURE_MEDIUM,
            color_map=self.method_colors
        )
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "r05_iqr_discriminatory_power.png",
            title="Discriminatory Power by Method"
        )
        plt.close(fig)
        
        return figure_path, table_path
    
    # ========================================================================
    # 4. TEMPORAL PERSISTENCE (YoY)
    # ========================================================================
    
    def visualize_yoy_persistence(self, rankings_dict: Dict[str, pd.DataFrame]) -> Tuple[Path, Path]:
        """
        Create year-to-year Spearman correlation plots + CSV export.
        
        Args:
            rankings_dict: Dict[method_name] -> DataFrame(provinces × years)
            
        Returns:
            Tuple of (figure_path, table_path)
        """
        yoy_correlations = {}
        
        for method, rankings_df in rankings_dict.items():
            yoy_rhos = []
            years = rankings_df.columns
            
            for i in range(len(years) - 1):
                year1, year2 = years[i], years[i + 1]
                rho, _ = spearmanr(rankings_df[year1], rankings_df[year2])
                yoy_rhos.append(rho)
            
            yoy_correlations[method] = yoy_rhos
        
        # Create DataFrame for CSV
        max_len = max(len(v) for v in yoy_correlations.values())
        yoy_df = pd.DataFrame({
            method: v + [np.nan] * (max_len - len(v))
            for method, v in yoy_correlations.items()
        })
        yoy_df.index = [f"{year1}-{year2}" for year1, year2 in 
                       zip(rankings_dict[list(rankings_dict.keys())[0]].columns[:-1],
                           rankings_dict[list(rankings_dict.keys())[0]].columns[1:])]
        
        # Save CSV
        table_path = save_dataframe_csv(
            yoy_df,
            self.tables_dir / "r04_yoy_persistence.csv",
            description="Year-to-year Spearman correlation of rankings"
        )
        
        # Create line plot
        fig = plot_multiline(
            yoy_df,
            title="Temporal Persistence of Rankings: Year-to-Year Spearman Correlation\n(Higher = More Stable Ranks)",
            xlabel="Year Transition",
            ylabel="Spearman ρ",
            figsize=PlotConfig.FIGURE_MEDIUM,
            color_map=self.method_colors,
            include_legend=True
        )
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "r06_yoy_temporal_persistence.png",
            title="Year-to-Year Ranking Persistence"
        )
        plt.close(fig)
        
        return figure_path, table_path
    
    # ========================================================================
    # 5. TOP-10 BUMP CHART
    # ========================================================================
    
    def visualize_top10_bump_chart(self, rankings_dict: Dict[str, pd.DataFrame]) -> Path:
        """
        Create bump chart for top-10 provinces across years.
        
        Note: Uses first method's rankings as reference (top 10 from that method).
        
        Args:
            rankings_dict: Dict[method_name] -> DataFrame(provinces × years)
            
        Returns:
            Figure path
        """
        # Get reference method (first one)
        ref_method = list(rankings_dict.keys())[0]
        ref_rankings = rankings_dict[ref_method]
        
        # Get top 10 provinces (by average rank)
        avg_ranks = ref_rankings.mean(axis=1)
        top_10_provinces = avg_ranks.nsmallest(10).index
        
        top_10_data = ref_rankings.loc[top_10_provinces]
        
        # Create bump chart
        fig, ax = plt.subplots(figsize=PlotConfig.FIGURE_LARGE, facecolor='white')
        
        years = top_10_data.columns
        colors = ColorPalette.get_viridis(len(top_10_provinces))
        
        for idx, (province, color) in enumerate(zip(top_10_data.index, colors)):
            ax.plot(
                years,
                top_10_data.loc[province],
                marker='o',
                markersize=8,
                linewidth=2.5,
                label=province,
                color=color,
                alpha=0.85
            )
        
        ax.set_title("Top 10 Provinces: Ranking Evolution (2011–2024)\n"
                    "Lower rank = Better performance",
                    fontsize=PlotConfig.FONT_SIZE_TITLE,
                    fontweight='bold',
                    pad=20)
        ax.set_xlabel("Year", fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
        ax.set_ylabel("Rank", fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
        ax.invert_yaxis()  # Lower ranks at top
        ax.set_ylim(0.5, 63.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                 frameon=True, shadow=True, fontsize=10, ncol=1)
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "r07_top10_bump_chart.png",
            title="Top 10 Provinces Bump Chart"
        )
        plt.close(fig)
        
        return figure_path
    
    # ========================================================================
    # ORCHESTRATION
    # ========================================================================
    
    def generate_all_visualizations(self,
                                   rankings_dict: Dict[str, pd.DataFrame],
                                   scores_dict: Dict[str, pd.DataFrame]) -> RankingVisualizationResult:
        """
        Generate all ranking visualizations and exports.
        
        Args:
            rankings_dict: Dict[method_name] -> DataFrame of ranks
            scores_dict: Dict[method_name] -> DataFrame of scores
            
        Returns:
            RankingVisualizationResult with all outputs
        """
        logger.info("=" * 70)
        logger.info("RANKING ANALYSIS: GENERATING VISUALIZATIONS & EXPORTS")
        logger.info("=" * 70)
        
        figures = {}
        tables = {}
        
        # 1. Ranking heatmaps (3 methods)
        heatmap_paths, table_path = self.visualize_ranking_heatmaps(rankings_dict)
        figures.update(heatmap_paths)
        tables['rankings'] = table_path
        
        # 2. Inter-method correlation
        fig_path, table_path = self.visualize_inter_method_correlation(rankings_dict)
        figures['inter_method_corr'] = fig_path
        tables['inter_method_corr'] = table_path
        
        # 3. Discriminatory power
        fig_path, table_path = self.visualize_discriminatory_power(scores_dict)
        figures['iqr_power'] = fig_path
        tables['iqr_power'] = table_path
        
        # 4. YoY persistence
        fig_path, table_path = self.visualize_yoy_persistence(rankings_dict)
        figures['yoy_persistence'] = fig_path
        tables['yoy_persistence'] = table_path
        
        # 5. Top 10 bump chart
        fig_path = self.visualize_top10_bump_chart(rankings_dict)
        figures['top10_bump'] = fig_path
        
        logger.info(f"\n✓ Generated {len(figures)} figures")
        logger.info(f"✓ Generated {len(tables)} tables")
        
        return RankingVisualizationResult(
            figures=figures,
            tables=tables,
            summary_metrics={
                'n_figures': len(figures),
                'n_tables': len(tables),
            }
        )
