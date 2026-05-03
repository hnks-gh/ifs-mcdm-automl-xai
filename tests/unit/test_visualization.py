"""Unit tests for Phase 10 visualization modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.utils.plot_utils import (
    PlotConfig, ColorPalette, save_dataframe_csv, save_figure,
    plot_heatmap, plot_multiline, plot_boxplot
)


class TestPlotConfig:
    """Tests for PlotConfig."""
    
    def test_plot_config_defaults(self):
        """Test PlotConfig default values."""
        assert PlotConfig.DPI == 300
        assert PlotConfig.FONT_SIZE_TITLE == 16
        assert PlotConfig.CMAP_SEQUENTIAL == "viridis"
    
    def test_method_colors_exist(self):
        """Test method colors are defined."""
        methods = ["if_waspas", "if_topsis", "if_promethee2"]
        for method in methods:
            assert method in PlotConfig.METHOD_COLORS


class TestColorPalette:
    """Tests for ColorPalette."""
    
    def test_viridis_palette(self):
        """Test Viridis palette generation."""
        colors_5 = ColorPalette.get_viridis(5)
        assert len(colors_5) == 5
        assert all(isinstance(c, tuple) for c in colors_5)
    
    def test_tab_safe_palette(self):
        """Test tab-safe palette generation."""
        colors_10 = ColorPalette.get_tab_safe(10)
        assert len(colors_10) == 10
        assert all(isinstance(c, str) for c in colors_10)
    
    def test_palette_length_mismatch(self):
        """Test palette with more colors requested than available."""
        colors = ColorPalette.get_tab_safe(15)
        assert len(colors) == 15


class TestSaveUtilities:
    """Tests for save functions."""
    
    def test_save_dataframe_csv(self):
        """Test saving DataFrame to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.5, 5.5, 6.5]})
            path = Path(tmpdir) / "test.csv"
            
            result = save_dataframe_csv(df, path, description="test data")
            
            assert result.exists()
            loaded = pd.read_csv(result, index_col=0)  # Read with index_col=0 since index is saved
            assert loaded.shape == (3, 2)
    
    def test_save_figure(self):
        """Test saving figure to PNG."""
        import matplotlib.pyplot as plt
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            path = Path(tmpdir) / "test_fig.png"
            
            result = save_figure(fig, path, title="test plot")
            
            assert result.exists()
            assert str(result).endswith('.png')
            plt.close(fig)


class TestPlottingFunctions:
    """Tests for plotting functions."""
    
    def test_heatmap_creation(self):
        """Test heatmap plotting."""
        import matplotlib.pyplot as plt
        
        data = pd.DataFrame(
            np.random.randn(5, 3),
            index=['A', 'B', 'C', 'D', 'E'],
            columns=['X', 'Y', 'Z']
        )
        
        fig = plot_heatmap(
            data, title="Test", xlabel="X", ylabel="Y",
            cbar_label="Value"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_multiline_plot(self):
        """Test multiline plotting."""
        import matplotlib.pyplot as plt
        
        data = pd.DataFrame({
            'Series1': [1, 2, 3, 4, 5],
            'Series2': [5, 4, 3, 2, 1],
        }, index=[2020, 2021, 2022, 2023, 2024])
        
        fig = plot_multiline(
            data, title="Test", xlabel="Year", ylabel="Value"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_boxplot_creation(self):
        """Test boxplot creation."""
        import matplotlib.pyplot as plt
        
        data = {
            'Method1': np.random.randn(100),
            'Method2': np.random.randn(100),
            'Method3': np.random.randn(100),
        }
        
        fig = plot_boxplot(data, title="Test", ylabel="Value")
        
        assert fig is not None
        plt.close(fig)


class TestWeightingVisualizer:
    """Tests for WeightingVisualizer."""
    
    def test_visualizer_initialization(self):
        """Test WeightingVisualizer initialization."""
        from src.mcdm.visualization import WeightingVisualizer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = WeightingVisualizer(Path(tmpdir))
            
            assert viz.figures_dir.exists()
            assert viz.tables_dir.exists()
    
    def test_criteria_weights_visualization(self):
        """Test criteria weights visualization."""
        from src.mcdm.visualization import WeightingVisualizer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = WeightingVisualizer(Path(tmpdir))
            
            # Create mock data
            criteria_weights = pd.DataFrame(
                np.random.rand(14, 8) * 0.2,
                index=range(2011, 2025),
                columns=[f'C{i:02d}' for i in range(1, 9)]
            )
            
            fig_path, table_path = viz.visualize_criteria_weights(criteria_weights)
            
            assert fig_path.exists()
            assert table_path.exists()
            
            import matplotlib.pyplot as plt
            plt.close('all')


class TestRankingVisualizer:
    """Tests for RankingVisualizer."""
    
    def test_visualizer_initialization(self):
        """Test RankingVisualizer initialization."""
        from src.mcdm.visualization import RankingVisualizer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = RankingVisualizer(Path(tmpdir))
            
            assert viz.figures_dir.exists()
            assert viz.tables_dir.exists()
    
    def test_ranking_heatmap_visualization(self):
        """Test ranking heatmap visualization."""
        from src.mcdm.visualization import RankingVisualizer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = RankingVisualizer(Path(tmpdir))
            
            # Create mock data
            rankings_dict = {
                'if_waspas': pd.DataFrame(
                    np.random.randint(1, 64, (63, 14)),
                    index=[f'P{i:02d}' for i in range(1, 64)],
                    columns=range(2011, 2025)
                ),
                'if_topsis': pd.DataFrame(
                    np.random.randint(1, 64, (63, 14)),
                    index=[f'P{i:02d}' for i in range(1, 64)],
                    columns=range(2011, 2025)
                ),
                'if_promethee2': pd.DataFrame(
                    np.random.randint(1, 64, (63, 14)),
                    index=[f'P{i:02d}' for i in range(1, 64)],
                    columns=range(2011, 2025)
                ),
            }
            
            heatmap_paths, table_path = viz.visualize_ranking_heatmaps(rankings_dict)
            
            assert len(heatmap_paths) > 0
            assert table_path.exists()
            
            import matplotlib.pyplot as plt
            plt.close('all')


class TestMLVisualizer:
    """Tests for MLVisualizer."""
    
    def test_visualizer_initialization(self):
        """Test MLVisualizer initialization."""
        from src.ml.visualization import MLVisualizer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MLVisualizer(Path(tmpdir))
            
            assert viz.figures_dir.exists()
            assert viz.tables_dir.exists()
    
    def test_forecast_visualization(self):
        """Test forecast visualization."""
        from src.ml.visualization import MLVisualizer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MLVisualizer(Path(tmpdir))
            
            # Create mock data
            forecast_df = pd.DataFrame(
                np.random.uniform(0, 3.33, (63, 29)),
                index=[f'P{i:02d}' for i in range(1, 64)],
                columns=[f'SC{i:02d}' for i in range(1, 30)]
            )
            
            fig_path, table_path = viz.visualize_forecast_2025(forecast_df)
            
            assert fig_path.exists()
            assert table_path.exists()
            
            import matplotlib.pyplot as plt
            plt.close('all')


class TestVisualizationOrchestrator:
    """Tests for VisualizationOrchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test VisualizationOrchestrator initialization."""
        from src.pipeline.visualization_orchestration import VisualizationOrchestrator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = VisualizationOrchestrator(Path(tmpdir))
            
            assert orchestrator.weighting_viz is not None
            assert orchestrator.ranking_viz is not None
            assert orchestrator.ml_viz is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
