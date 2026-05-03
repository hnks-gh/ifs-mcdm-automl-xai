"""
ML Forecasting & SHAP Explainability Visualization & Export Module

Generates publication-quality figures and comprehensive data tables for:
1. MICE imputation summary (before/after statistics)
2. 2025 forecast heatmaps and summary statistics
3. SHAP global feature importance rankings
4. SHAP beeswarm plots (top features)
5. SHAP waterfall plots (top 5 provinces per target)

Outputs:
- Figures: PNG at 300 DPI (5+ visualizations)
- Tables: CSV format with full precision and metadata
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import logging
from dataclasses import dataclass

from ...utils.plot_utils import (
    PlotConfig, ColorPalette, set_publication_style,
    save_figure, save_dataframe_csv,
    plot_heatmap, plot_horizontal_bars, plot_before_after_comparison
)

logger = logging.getLogger(__name__)


@dataclass
class MLVisualizationResult:
    """Container for ML visualization outputs."""
    figures: Dict[str, Path]
    tables: Dict[str, Path]
    summary_metrics: Dict[str, Any]


class MLVisualizer:
    """Generates all ML analysis visualizations and exports."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Base output directory (e.g., 'output/')
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures" / "ml"
        self.tables_dir = self.output_dir / "tables" / "ml"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        set_publication_style()
    
    # ========================================================================
    # 1. IMPUTATION SUMMARY
    # ========================================================================
    
    def visualize_imputation_summary(self, before_stats: pd.DataFrame,
                                    after_stats: pd.DataFrame) -> Tuple[Path, Path]:
        """
        Create before/after imputation comparison + CSV export.
        
        Args:
            before_stats: DataFrame with missing data statistics before imputation
            after_stats: DataFrame with statistics after imputation
            
        Returns:
            Tuple of (figure_path, table_path)
        """
        # Prepare combined table
        summary_df = pd.DataFrame({
            'Before Imputation': before_stats.sum(),
            'After Imputation': after_stats.sum(),
        })
        
        # Add percentage improvement
        summary_df['Improvement %'] = (
            (before_stats.sum() - after_stats.sum()) / before_stats.sum() * 100
        )
        
        # Save CSV
        table_path = save_dataframe_csv(
            summary_df,
            self.tables_dir / "m01_imputation_summary.csv",
            description="MICE imputation summary statistics"
        )
        
        # Create visualization
        fig = plot_before_after_comparison(
            before_stats.sum(),
            after_stats.sum(),
            title="MICE Imputation Impact: Missing Values (Before vs After)",
            ylabel="Number of Missing Values",
            figsize=PlotConfig.FIGURE_MEDIUM
        )
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "m01_imputation_summary.png",
            title="Imputation Summary"
        )
        plt.close(fig)
        
        return figure_path, table_path
    
    # ========================================================================
    # 2. 2025 FORECAST HEATMAP
    # ========================================================================
    
    def visualize_forecast_2025(self, forecast_df: pd.DataFrame) -> Tuple[Path, Path]:
        """
        Create heatmap of 2025 forecasted values + CSV export.
        
        Args:
            forecast_df: DataFrame shape (63 provinces × 29 sub-criteria)
            
        Returns:
            Tuple of (figure_path, table_path)
        """
        # Ensure numeric
        forecast_df = forecast_df.apply(pd.to_numeric, errors='coerce')
        
        # Save CSV with full precision
        table_path = save_dataframe_csv(
            forecast_df,
            self.tables_dir / "m02_forecast_2025.csv",
            description="2025 forecasted values for all sub-criteria and provinces"
        )
        
        # Create heatmap
        fig = plot_heatmap(
            forecast_df,
            title="AutoGluon Multivariate Time Series Forecast: Vietnam PAPI 2025\n(63 Provinces × 29 Sub-Criteria)",
            xlabel="Sub-Criteria",
            ylabel="Province",
            figsize=(16, 18),
            cmap="viridis",
            cbar_label="Forecasted Score",
            annot=False,  # Too dense for annotations
        )
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "m02_forecast_2025_heatmap.png",
            title="2025 Forecast Heatmap"
        )
        plt.close(fig)
        
        return figure_path, table_path
    
    # ========================================================================
    # 3. FORECAST STATISTICS
    # ========================================================================
    
    def visualize_forecast_statistics(self, forecast_df: pd.DataFrame,
                                     historical_df: pd.DataFrame) -> Tuple[Path, Path]:
        """
        Create statistics comparing 2025 forecasts with historical data + CSV export.
        
        Args:
            forecast_df: 2025 forecast DataFrame
            historical_df: Historical data (e.g., 2024) for comparison
            
        Returns:
            Tuple of (figure_path, table_path)
        """
        # Compute descriptive statistics
        stats_df = pd.DataFrame({
            '2025 Forecast - Mean': forecast_df.mean(),
            '2025 Forecast - Std': forecast_df.std(),
            '2024 Historical - Mean': historical_df.mean(),
            '2024 Historical - Std': historical_df.std(),
            'Mean Change': forecast_df.mean() - historical_df.mean(),
        })
        
        # Save CSV
        table_path = save_dataframe_csv(
            stats_df,
            self.tables_dir / "m03_forecast_statistics.csv",
            description="2025 forecast statistics compared to 2024 historical data"
        )
        
        # Create visualization comparing means
        fig, ax = plt.subplots(figsize=PlotConfig.FIGURE_WIDE, facecolor='white')
        
        x = np.arange(len(stats_df))
        width = 0.35
        
        ax.bar(x - width/2, stats_df['2024 Historical - Mean'], width,
               label='2024 Historical', color='#1f77b4', alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, stats_df['2025 Forecast - Mean'], width,
               label='2025 Forecast', color='#ff7f0e', alpha=0.7, edgecolor='black')
        
        ax.set_title("Mean Score Comparison: 2024 Historical vs 2025 Forecast\n(All Sub-Criteria)",
                    fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', pad=20)
        ax.set_xlabel("Sub-Criteria", fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
        ax.set_ylabel("Mean Score", fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df.index, rotation=45, ha='right')
        ax.legend(frameon=True, shadow=True, fontsize=PlotConfig.FONT_SIZE_LEGEND)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "m02b_forecast_statistics.png",
            title="Forecast Statistics Comparison"
        )
        plt.close(fig)
        
        return figure_path, table_path
    
    # ========================================================================
    # 4. SHAP GLOBAL IMPORTANCE
    # ========================================================================
    
    def visualize_shap_global_importance(self, shap_importance_dict: Dict[str, float],
                                        top_n: int = 15) -> Tuple[Path, Path]:
        """
        Create horizontal bar chart of SHAP-based feature importance + CSV export.
        
        Args:
            shap_importance_dict: Dict mapping feature names to mean |SHAP| values
            top_n: Number of top features to display
            
        Returns:
            Tuple of (figure_path, table_path)
        """
        # Convert to Series and sort
        importance_series = pd.Series(shap_importance_dict).sort_values(ascending=False)
        
        # Prepare full table for CSV
        importance_df = pd.DataFrame({
            'Feature': importance_series.index,
            'Mean |SHAP|': importance_series.values,
            'Rank': range(1, len(importance_series) + 1)
        }).set_index('Feature')
        
        # Save CSV
        table_path = save_dataframe_csv(
            importance_df,
            self.tables_dir / "m04_shap_importance.csv",
            description="SHAP-based global feature importance ranking"
        )
        
        # Create bar plot (top N)
        top_importance = importance_series.head(top_n)
        
        fig, ax = plt.subplots(figsize=PlotConfig.FIGURE_MEDIUM, facecolor='white')
        
        colors = ColorPalette.get_viridis(len(top_importance))
        y_pos = np.arange(len(top_importance))
        
        ax.barh(y_pos, top_importance.values, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_importance.index)
        
        ax.set_title(f"SHAP Global Feature Importance: Top {top_n} Features\n"
                    "Mean Absolute SHAP Value across all predictions",
                    fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', pad=20)
        ax.set_xlabel("Mean |SHAP| Value", fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, v in enumerate(top_importance.values):
            ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "m03_shap_global_importance.png",
            title="SHAP Global Feature Importance"
        )
        plt.close(fig)
        
        return figure_path, table_path
    
    # ========================================================================
    # 5. SHAP BEESWARM PLOTS
    # ========================================================================
    
    def visualize_shap_beeswarm(self, shap_values_dict: Dict[str, np.ndarray],
                               feature_names: List[str],
                               top_n_features: int = 10) -> Path:
        """
        Create SHAP beeswarm plot for top features.
        
        Args:
            shap_values_dict: Dict mapping feature names to SHAP value arrays
            feature_names: List of feature names
            top_n_features: Number of top features to plot
            
        Returns:
            Figure path
        """
        # Compute mean absolute SHAP values
        mean_shap = {
            fname: np.mean(np.abs(shap_values_dict[fname]))
            for fname in feature_names if fname in shap_values_dict
        }
        
        # Get top features
        top_features = sorted(mean_shap.items(), key=lambda x: x[1], reverse=True)[:top_n_features]
        top_names = [f[0] for f in top_features]
        
        # Create plot
        fig, ax = plt.subplots(figsize=PlotConfig.FIGURE_MEDIUM, facecolor='white')
        
        y_pos = np.arange(len(top_names))
        colors = ColorPalette.get_viridis(len(top_names))
        
        for i, feature in enumerate(top_names):
            shap_vals = shap_values_dict.get(feature, np.array([]))
            if len(shap_vals) > 0:
                # Add jitter for visibility
                jittered_x = shap_vals + np.random.normal(0, 0.02, len(shap_vals))
                ax.scatter(jittered_x, np.full_like(jittered_x, i),
                          alpha=0.6, s=20, color=colors[i])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names)
        ax.set_xlabel("SHAP Value", fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
        ax.set_title(f"SHAP Beeswarm Plot: Top {top_n_features} Features\n"
                    "Each dot = one prediction's SHAP value",
                    fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', pad=20)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "m04_shap_beeswarm_top10.png",
            title="SHAP Beeswarm Plot"
        )
        plt.close(fig)
        
        return figure_path
    
    # ========================================================================
    # 6. SHAP WATERFALL PLOTS
    # ========================================================================
    
    def visualize_shap_waterfall_samples(self, forecast_df: pd.DataFrame,
                                        shap_values_df: pd.DataFrame,
                                        base_value: float,
                                        n_samples: int = 5) -> Path:
        """
        Create waterfall plots for top N provinces.
        
        Args:
            forecast_df: Forecast values DataFrame
            shap_values_df: SHAP values DataFrame (same shape as forecast_df)
            base_value: Base model prediction value
            n_samples: Number of samples (provinces) to plot
            
        Returns:
            Figure path
        """
        # Select top n provinces by absolute SHAP contribution
        shap_contribution = np.abs(shap_values_df).mean(axis=1)
        top_indices = np.argsort(shap_contribution)[-n_samples:]
        top_provinces = forecast_df.index[top_indices]
        
        # Create subplots
        nrows = int(np.ceil(n_samples / 2))
        fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows), facecolor='white')
        axes = axes.flatten()
        
        for idx, (ax, province) in enumerate(zip(axes, top_provinces)):
            # Get SHAP values for this province
            shap_vals = shap_values_df.loc[province].values
            forecast_vals = forecast_df.loc[province].values
            
            # Sort by absolute SHAP value
            sort_idx = np.argsort(np.abs(shap_vals))[-10:]  # Top 10 features
            
            feature_names = shap_values_df.columns[sort_idx]
            shap_sorted = shap_vals[sort_idx]
            
            # Create cumulative effect
            cumulative = base_value + np.cumsum(shap_sorted)
            
            # Plot
            colors = ['#ff7f0e' if x > 0 else '#1f77b4' for x in shap_sorted]
            
            ax.barh(range(len(feature_names)), shap_sorted, color=colors, alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names, fontsize=9)
            ax.axvline(x=0, color='black', linewidth=1)
            ax.set_title(f"Waterfall: {province} | Base: {base_value:.3f} → Prediction: {np.sum(shap_sorted) + base_value:.3f}",
                        fontsize=10, fontweight='bold')
            ax.set_xlabel("SHAP Value", fontsize=9)
            ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Hide empty subplots
        for idx in range(len(top_provinces), len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f"SHAP Waterfall Plots: Top {n_samples} Provinces by SHAP Contribution",
                    fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', y=0.995)
        
        figure_path = save_figure(
            fig,
            self.figures_dir / "m05_shap_waterfall_top5.png",
            title="SHAP Waterfall Plots"
        )
        plt.close(fig)
        
        return figure_path
    
    # ========================================================================
    # ORCHESTRATION
    # ========================================================================
    
    def generate_all_visualizations(self,
                                   before_imputation_stats: pd.DataFrame,
                                   after_imputation_stats: pd.DataFrame,
                                   forecast_2025: pd.DataFrame,
                                   historical_2024: pd.DataFrame,
                                   shap_importance: Dict[str, float],
                                   shap_values_dict: Optional[Dict[str, np.ndarray]] = None,
                                   forecast_with_shap: Optional[pd.DataFrame] = None,
                                   shap_with_values: Optional[pd.DataFrame] = None,
                                   base_value: float = 0.0) -> MLVisualizationResult:
        """
        Generate all ML visualizations and exports.
        
        Args:
            before_imputation_stats: Pre-imputation statistics
            after_imputation_stats: Post-imputation statistics
            forecast_2025: 2025 forecast DataFrame
            historical_2024: 2024 historical data for comparison
            shap_importance: Dict mapping features to importance scores
            shap_values_dict: Optional dict of SHAP value arrays
            forecast_with_shap: Optional forecast data for waterfall plots
            shap_with_values: Optional SHAP values for waterfall plots
            base_value: Base value for waterfall plots
            
        Returns:
            MLVisualizationResult with all outputs
        """
        logger.info("=" * 70)
        logger.info("ML ANALYSIS: GENERATING VISUALIZATIONS & EXPORTS")
        logger.info("=" * 70)
        
        figures = {}
        tables = {}
        
        # 1. Imputation summary
        fig_path, table_path = self.visualize_imputation_summary(
            before_imputation_stats, after_imputation_stats
        )
        figures['imputation_summary'] = fig_path
        tables['imputation_summary'] = table_path
        
        # 2. Forecast heatmap
        fig_path, table_path = self.visualize_forecast_2025(forecast_2025)
        figures['forecast_heatmap'] = fig_path
        tables['forecast_2025'] = table_path
        
        # 3. Forecast statistics
        fig_path, table_path = self.visualize_forecast_statistics(
            forecast_2025, historical_2024
        )
        figures['forecast_stats'] = fig_path
        tables['forecast_stats'] = table_path
        
        # 4. SHAP global importance
        fig_path, table_path = self.visualize_shap_global_importance(shap_importance)
        figures['shap_importance'] = fig_path
        tables['shap_importance'] = table_path
        
        # 5. SHAP beeswarm (if data available)
        if shap_values_dict:
            fig_path = self.visualize_shap_beeswarm(
                shap_values_dict,
                list(shap_values_dict.keys())
            )
            figures['shap_beeswarm'] = fig_path
        
        # 6. SHAP waterfall (if data available)
        if forecast_with_shap is not None and shap_with_values is not None:
            fig_path = self.visualize_shap_waterfall_samples(
                forecast_with_shap, shap_with_values, base_value
            )
            figures['shap_waterfall'] = fig_path
        
        logger.info(f"\n✓ Generated {len(figures)} figures")
        logger.info(f"✓ Generated {len(tables)} tables")
        
        return MLVisualizationResult(
            figures=figures,
            tables=tables,
            summary_metrics={
                'n_figures': len(figures),
                'n_tables': len(tables),
            }
        )
