"""
Unified visualization utilities for publication-quality figures.

Provides:
- Professional styling and color palettes (colorblind-safe, scientific)
- Common plot templates and helpers
- High-resolution PNG export (300 DPI)
- Consistent typography and layout
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# STYLING CONFIGURATION
# ============================================================================

class PlotConfig:
    """Publication-quality plot configuration."""
    
    # Resolution and sizes
    DPI = 300
    FIGURE_SMALL = (12, 8)
    FIGURE_MEDIUM = (14, 9)
    FIGURE_LARGE = (16, 10)
    FIGURE_WIDE = (18, 8)
    FIGURE_SQUARE = (10, 10)
    
    # Typography
    FONT_FAMILY = "sans-serif"
    FONT_SIZE_TITLE = 16
    FONT_SIZE_LABEL = 12
    FONT_SIZE_TICK = 11
    FONT_SIZE_LEGEND = 10
    
    # Colors - Viridis (scientific, colorblind-safe)
    CMAP_SEQUENTIAL = "viridis"
    CMAP_DIVERGING = "RdBu_r"
    
    # Method-specific colors (colorblind-safe palette)
    METHOD_COLORS = {
        "if_waspas": "#1f77b4",      # Blue
        "if_topsis": "#ff7f0e",      # Orange
        "if_promethee2": "#2ca02c",  # Green
        "waspas": "#1f77b4",
        "topsis": "#ff7f0e",
        "promethee": "#2ca02c",
    }
    
    # Line and marker styles
    LINEWIDTH_PLOT = 2.0
    LINEWIDTH_AXIS = 1.5
    MARKERSIZE = 8
    HATCH_PATTERNS = ['///', '\\\\\\', '|||', '---', '+++', 'xxx']
    
    # Layout
    TIGHT_LAYOUT_PAD = 1.5


class ColorPalette:
    """Colorblind-safe, scientific color palettes."""
    
    # Viridis: excellent for all forms of color blindness
    @staticmethod
    def get_viridis(n: int = 10) -> List[str]:
        """Get n colors from Viridis palette."""
        return [plt.cm.viridis(i / (n - 1)) if n > 1 else plt.cm.viridis(0.5) 
                for i in range(n)]
    
    # Tab20 with accessibility enhancements
    @staticmethod
    def get_tab_safe(n: int = 10) -> List[str]:
        """Get colorblind-safe tab palette."""
        safe_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        if n <= len(safe_colors):
            return safe_colors[:n]
        return [plt.cm.tab20(i / (n - 1)) for i in range(n)]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_publication_style():
    """Apply publication-quality styling globally."""
    plt.rcParams.update({
        'font.family': PlotConfig.FONT_FAMILY,
        'font.size': PlotConfig.FONT_SIZE_LABEL,
        'axes.labelsize': PlotConfig.FONT_SIZE_LABEL,
        'axes.titlesize': PlotConfig.FONT_SIZE_TITLE,
        'xtick.labelsize': PlotConfig.FONT_SIZE_TICK,
        'ytick.labelsize': PlotConfig.FONT_SIZE_TICK,
        'legend.fontsize': PlotConfig.FONT_SIZE_LEGEND,
        'lines.linewidth': PlotConfig.LINEWIDTH_PLOT,
        'axes.linewidth': PlotConfig.LINEWIDTH_AXIS,
        'xtick.major.width': PlotConfig.LINEWIDTH_AXIS,
        'ytick.major.width': PlotConfig.LINEWIDTH_AXIS,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def save_figure(fig: plt.Figure, output_path: Path, title: Optional[str] = None,
                tight_layout: bool = True, dpi: int = PlotConfig.DPI) -> Path:
    """
    Save figure with high resolution and metadata.
    
    Args:
        fig: Matplotlib figure object
        output_path: Path to save PNG
        title: Optional title for logging
        tight_layout: Whether to apply tight layout
        dpi: Resolution in DPI
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if tight_layout:
        fig.tight_layout(pad=PlotConfig.TIGHT_LAYOUT_PAD)
    
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
    )
    
    logger.info(f"✓ Saved figure: {output_path.name} ({title or 'no title'})")
    return output_path


def save_dataframe_csv(df: pd.DataFrame, output_path: Path, 
                       description: Optional[str] = None,
                       index: bool = True) -> Path:
    """
    Save DataFrame as CSV with consistent formatting.
    
    Args:
        df: DataFrame to save
        output_path: Path to save CSV
        description: Optional description for logging
        index: Whether to include index
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=index, float_format='%.6g')
    
    logger.info(f"✓ Saved table: {output_path.name} {description or ''} "
                f"({df.shape[0]}x{df.shape[1]})")
    return output_path


# ============================================================================
# HEATMAP UTILITIES
# ============================================================================

def plot_heatmap(data: pd.DataFrame, title: str, xlabel: str, ylabel: str,
                 figsize: Tuple[int, int] = PlotConfig.FIGURE_MEDIUM,
                 cmap: str = PlotConfig.CMAP_SEQUENTIAL,
                 cbar_label: Optional[str] = None,
                 annot: bool = True, fmt: str = '.3f',
                 vmin: Optional[float] = None, vmax: Optional[float] = None) -> plt.Figure:
    """
    Create publication-quality heatmap.
    
    Args:
        data: 2D DataFrame for heatmap
        title: Figure title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        cmap: Colormap name
        cbar_label: Colorbar label
        annot: Whether to annotate cells
        fmt: Format string for annotations
        vmin/vmax: Data range for normalization
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        cbar_kws={'label': cbar_label or 'Value'},
        annot=annot,
        fmt=fmt,
        linewidths=0.5,
        linecolor='gray',
        vmin=vmin,
        vmax=vmax,
        cbar=True
    )
    
    ax.set_title(title, fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
    
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str,
                             figsize: Tuple[int, int] = PlotConfig.FIGURE_SQUARE,
                             annot: bool = True, fmt: str = '.3f') -> plt.Figure:
    """
    Create correlation matrix heatmap with diverging colormap.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Figure title
        figsize: Figure size
        annot: Whether to annotate cells
        fmt: Format string for annotations
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap=PlotConfig.CMAP_DIVERGING,
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        annot=annot,
        fmt=fmt,
        linewidths=1.0,
        linecolor='white',
        square=True,
        cbar_kws={'label': 'Correlation coefficient'}
    )
    
    ax.set_title(title, fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', pad=20)
    
    return fig


# ============================================================================
# LINE PLOT UTILITIES
# ============================================================================

def plot_multiline(data: pd.DataFrame, title: str, xlabel: str, ylabel: str,
                   figsize: Tuple[int, int] = PlotConfig.FIGURE_MEDIUM,
                   color_map: Optional[Dict[str, str]] = None,
                   include_legend: bool = True) -> plt.Figure:
    """
    Create multi-line plot with consistent styling.
    
    Args:
        data: DataFrame with data series (columns = lines)
        title: Figure title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        color_map: Dict mapping column names to colors
        include_legend: Whether to show legend
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    colors = ColorPalette.get_viridis(len(data.columns))
    
    for i, col in enumerate(data.columns):
        color = color_map.get(col, colors[i]) if color_map else colors[i]
        ax.plot(
            data.index,
            data[col],
            marker='o',
            label=col,
            linewidth=PlotConfig.LINEWIDTH_PLOT,
            markersize=PlotConfig.MARKERSIZE,
            color=color,
            alpha=0.8
        )
    
    ax.set_title(title, fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if include_legend:
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=PlotConfig.FONT_SIZE_LEGEND)
    
    return fig


# ============================================================================
# BAR PLOT UTILITIES
# ============================================================================

def plot_grouped_bars(data: pd.DataFrame, title: str, xlabel: str, ylabel: str,
                      figsize: Tuple[int, int] = PlotConfig.FIGURE_MEDIUM,
                      color_map: Optional[Dict[str, str]] = None) -> plt.Figure:
    """
    Create grouped bar chart with consistent styling.
    
    Args:
        data: DataFrame (rows = groups, columns = categories)
        title: Figure title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        color_map: Dict mapping column names to colors
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    x = np.arange(len(data.index))
    width = 0.8 / len(data.columns)
    
    colors = ColorPalette.get_viridis(len(data.columns))
    
    for i, col in enumerate(data.columns):
        offset = width * (i - len(data.columns) / 2 + 0.5)
        color = color_map.get(col, colors[i]) if color_map else colors[i]
        ax.bar(
            x + offset,
            data[col],
            width,
            label=col,
            color=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.85
        )
    
    ax.set_title(title, fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=45, ha='right')
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=PlotConfig.FONT_SIZE_LEGEND)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    return fig


def plot_horizontal_bars(data: pd.Series, title: str, xlabel: str,
                        figsize: Tuple[int, int] = PlotConfig.FIGURE_MEDIUM,
                        color: str = '#1f77b4', top_n: Optional[int] = None) -> plt.Figure:
    """
    Create horizontal bar chart (ideal for feature importance).
    
    Args:
        data: Series with values
        title: Figure title
        xlabel: X-axis label
        figsize: Figure size
        color: Bar color
        top_n: Show only top N features
    
    Returns:
        Matplotlib figure object
    """
    if top_n:
        data = data.nlargest(top_n)
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Sort by value
    data = data.sort_values(ascending=True)
    
    colors = ColorPalette.get_viridis(len(data))
    ax.barh(range(len(data)), data.values, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index)
    ax.set_title(title, fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    return fig


# ============================================================================
# BOX PLOT UTILITIES
# ============================================================================

def plot_boxplot(data: Dict[str, np.ndarray], title: str, ylabel: str,
                 figsize: Tuple[int, int] = PlotConfig.FIGURE_MEDIUM,
                 color_map: Optional[Dict[str, str]] = None) -> plt.Figure:
    """
    Create boxplot with consistent styling.
    
    Args:
        data: Dict mapping labels to data arrays
        title: Figure title
        ylabel: Y-axis label
        figsize: Figure size
        color_map: Dict mapping labels to colors
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    labels = list(data.keys())
    values = list(data.values())
    
    colors = ColorPalette.get_viridis(len(labels))
    bp = ax.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=6)
    )
    
    for patch, label in zip(bp['boxes'], labels):
        color = color_map.get(label, colors[labels.index(label)]) if color_map else colors[labels.index(label)]
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title(title, fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    return fig


# ============================================================================
# RADAR PLOT UTILITIES
# ============================================================================

def plot_radar_grid(data_dict: Dict[str, Dict[str, float]], title: str,
                   figsize: Tuple[int, int] = (16, 14),
                   ncols: int = 4) -> plt.Figure:
    """
    Create grid of radar plots (ideal for multi-year weight profiles).
    
    Args:
        data_dict: Dict[name] -> Dict[axis_name] -> value
        title: Figure title
        figsize: Figure size
        ncols: Number of columns in grid
    
    Returns:
        Matplotlib figure object
    """
    n_plots = len(data_dict)
    nrows = int(np.ceil(n_plots / ncols))
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        subplot_kw=dict(projection='polar'),
        facecolor='white'
    )
    
    if n_plots == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.reshape(nrows, ncols)
    
    axes_flat = axes.flatten()
    
    for idx, (name, data) in enumerate(data_dict.items()):
        ax = axes_flat[idx]
        
        categories = list(data.keys())
        values = list(data.values())
        
        # Close the plot
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles_plot = angles + [angles[0]]
        
        ax.plot(angles_plot, values_plot, 'o-', linewidth=2, color='#1f77b4', alpha=0.7)
        ax.fill(angles_plot, values_plot, alpha=0.25, color='#1f77b4')
        
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, max(values) * 1.1)
        ax.set_title(name, fontsize=11, fontweight='bold', pad=20)
        ax.grid(True)
    
    # Hide empty subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    fig.suptitle(title, fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', y=0.98)
    
    return fig


# ============================================================================
# COMBINATION PLOTS
# ============================================================================

def plot_before_after_comparison(before_data: pd.Series, after_data: pd.Series,
                                title: str, ylabel: str,
                                figsize: Tuple[int, int] = PlotConfig.FIGURE_MEDIUM) -> plt.Figure:
    """
    Create before/after comparison plot.
    
    Args:
        before_data: Series with before values
        after_data: Series with after values
        title: Figure title
        ylabel: Y-axis label
        figsize: Figure size
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    x = np.arange(len(before_data))
    width = 0.35
    
    ax.bar(x - width/2, before_data.values, width, label='Before', color='#d62728', alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, after_data.values, width, label='After', color='#2ca02c', alpha=0.7, edgecolor='black')
    
    ax.set_title(title, fontsize=PlotConfig.FONT_SIZE_TITLE, fontweight='bold', pad=20)
    ax.set_xlabel('Category', fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=PlotConfig.FONT_SIZE_LABEL, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(before_data.index, rotation=45, ha='right')
    ax.legend(frameon=True, shadow=True, fontsize=PlotConfig.FONT_SIZE_LEGEND)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    return fig


# ============================================================================
# Initialize styling on import
# ============================================================================

set_publication_style()
