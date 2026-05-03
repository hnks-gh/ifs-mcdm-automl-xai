"""Integration tests for Phase 10 visualization orchestration."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.pipeline.visualization_orchestration import (
    VisualizationOrchestrator, orchestrate_phase10_visualizations
)


class TestVisualizationIntegration:
    """Integration tests for Phase 10 visualization system."""
    
    @pytest.fixture
    def mock_mcdm_data(self):
        """Create mock MCDM analysis data."""
        criteria_weights = pd.DataFrame(
            np.random.rand(14, 8) * 0.2,
            index=range(2011, 2025),
            columns=[f'C{i:02d}' for i in range(1, 9)]
        )
        
        subcriteria_weights = pd.DataFrame(
            np.random.rand(14, 29) * 0.1,
            index=range(2011, 2025),
            columns=[f'SC{i:02d}' for i in range(11, 84)]
        )
        
        rankings_dict = {
            'if_waspas': pd.DataFrame(
                np.argsort(np.random.rand(63, 14), axis=0) + 1,
                index=[f'P{i:02d}' for i in range(1, 64)],
                columns=range(2011, 2025)
            ),
            'if_topsis': pd.DataFrame(
                np.argsort(np.random.rand(63, 14), axis=0) + 1,
                index=[f'P{i:02d}' for i in range(1, 64)],
                columns=range(2011, 2025)
            ),
            'if_promethee2': pd.DataFrame(
                np.argsort(np.random.rand(63, 14), axis=0) + 1,
                index=[f'P{i:02d}' for i in range(1, 64)],
                columns=range(2011, 2025)
            ),
        }
        
        scores_dict = {
            'if_waspas': pd.DataFrame(
                np.random.uniform(0, 1, (63, 14)),
                index=[f'P{i:02d}' for i in range(1, 64)],
                columns=range(2011, 2025)
            ),
            'if_topsis': pd.DataFrame(
                np.random.uniform(0, 1, (63, 14)),
                index=[f'P{i:02d}' for i in range(1, 64)],
                columns=range(2011, 2025)
            ),
            'if_promethee2': pd.DataFrame(
                np.random.uniform(0, 1, (63, 14)),
                index=[f'P{i:02d}' for i in range(1, 64)],
                columns=range(2011, 2025)
            ),
        }
        
        stability_results = {
            'rmsd': np.random.uniform(0, 0.1, 10),
            'cv': {
                f'SC{i:02d}': {'mean': np.random.uniform(0.1, 0.3), 'std': np.random.uniform(0.01, 0.05)}
                for i in range(11, 84)
            },
            'subcriteria': [f'SC{i:02d}' for i in range(11, 84)],
            'windows': [[2011+i, 2015+i] for i in range(10)],
        }
        
        sensitivity_results = {
            'if_waspas': np.random.uniform(0.7, 1.0, 10000),
            'if_topsis': np.random.uniform(0.7, 1.0, 10000),
            'if_promethee2': np.random.uniform(0.7, 1.0, 10000),
        }
        
        return {
            'criteria_weights': criteria_weights,
            'subcriteria_weights': subcriteria_weights,
            'rankings_dict': rankings_dict,
            'scores_dict': scores_dict,
            'stability_results': stability_results,
            'sensitivity_results': sensitivity_results,
        }
    
    @pytest.fixture
    def mock_ml_data(self):
        """Create mock ML analysis data."""
        before_stats = pd.DataFrame(
            np.random.randint(0, 100, (10, 3)),
            columns=['Missing Cells', 'Missing Rows', 'Missing Percentage']
        )
        
        after_stats = pd.DataFrame(
            np.zeros((10, 3)),
            columns=['Missing Cells', 'Missing Rows', 'Missing Percentage']
        )
        
        forecast_2025 = pd.DataFrame(
            np.random.uniform(0, 3.33, (63, 29)),
            index=[f'P{i:02d}' for i in range(1, 64)],
            columns=[f'SC{i:02d}' for i in range(11, 40)]
        )
        
        historical_2024 = pd.DataFrame(
            np.random.uniform(0, 3.33, (63, 29)),
            index=[f'P{i:02d}' for i in range(1, 64)],
            columns=[f'SC{i:02d}' for i in range(11, 40)]
        )
        
        shap_importance = {
            f'SC{i:02d}': np.random.uniform(0, 0.5)
            for i in range(11, 40)
        }
        
        return {
            'before_imputation_stats': before_stats,
            'after_imputation_stats': after_stats,
            'forecast_2025': forecast_2025,
            'historical_2024': historical_2024,
            'shap_importance': shap_importance,
        }
    
    def test_orchestrator_mcdm_generation(self, mock_mcdm_data):
        """Test MCDM visualization generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = VisualizationOrchestrator(Path(tmpdir))
            
            results = orchestrator.generate_mcdm_visualizations(**mock_mcdm_data)
            
            assert 'weighting' in results
            assert 'ranking' in results
            
            weighting_result = results['weighting']
            ranking_result = results['ranking']
            
            assert len(weighting_result.figures) > 0
            assert len(weighting_result.tables) > 0
            assert len(ranking_result.figures) > 0
            assert len(ranking_result.tables) > 0
    
    def test_orchestrator_ml_generation(self, mock_ml_data):
        """Test ML visualization generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = VisualizationOrchestrator(Path(tmpdir))
            
            results = orchestrator.generate_ml_visualizations(**mock_ml_data)
            
            assert 'ml' in results
            
            ml_result = results['ml']
            
            assert len(ml_result.figures) > 0
            assert len(ml_result.tables) > 0
    
    def test_manifest_generation(self, mock_mcdm_data, mock_ml_data):
        """Test manifest generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = VisualizationOrchestrator(Path(tmpdir))
            
            mcdm_results = orchestrator.generate_mcdm_visualizations(**mock_mcdm_data)
            ml_results = orchestrator.generate_ml_visualizations(**mock_ml_data)
            
            summary = orchestrator.generate_visualization_manifest(mcdm_results, ml_results)
            
            assert summary.total_figures > 0
            assert summary.total_tables > 0
            assert summary.mcdm_figures > 0
            assert summary.ml_figures > 0
    
    def test_validation_report(self, mock_mcdm_data, mock_ml_data):
        """Test validation report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = VisualizationOrchestrator(Path(tmpdir))
            
            mcdm_results = orchestrator.generate_mcdm_visualizations(**mock_mcdm_data)
            ml_results = orchestrator.generate_ml_visualizations(**mock_ml_data)
            
            summary = orchestrator.generate_visualization_manifest(mcdm_results, ml_results)
            validation = orchestrator.validate_outputs(summary)
            
            report_path = orchestrator.generate_validation_report(summary, validation)
            
            assert report_path.exists()
            assert report_path.suffix == '.csv'
    
    def test_orchestrate_function(self, mock_mcdm_data, mock_ml_data):
        """Test convenience orchestration function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = orchestrate_phase10_visualizations(
                output_base_dir=Path(tmpdir),
                mcdm_data=mock_mcdm_data,
                ml_data=mock_ml_data,
            )
            
            assert summary.total_figures > 0
            assert summary.total_tables > 0
            
            # Check manifest files
            manifest_dir = Path(tmpdir) / "manifests"
            assert manifest_dir.exists()


class TestOutputStructure:
    """Tests for output directory structure."""
    
    def test_output_directory_creation(self):
        """Test output directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = VisualizationOrchestrator(Path(tmpdir))
            
            # Directories should be created
            assert (Path(tmpdir) / "figures").exists()
            assert (Path(tmpdir) / "tables").exists()
    
    def test_figures_organization(self):
        """Test figures are organized by category."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = VisualizationOrchestrator(Path(tmpdir))
            
            # Create mock figure structure
            (Path(tmpdir) / "figures" / "weighting").mkdir(parents=True, exist_ok=True)
            (Path(tmpdir) / "figures" / "ranking").mkdir(parents=True, exist_ok=True)
            (Path(tmpdir) / "figures" / "ml").mkdir(parents=True, exist_ok=True)
            
            assert (Path(tmpdir) / "figures" / "weighting").exists()
            assert (Path(tmpdir) / "figures" / "ranking").exists()
            assert (Path(tmpdir) / "figures" / "ml").exists()
    
    def test_tables_organization(self):
        """Test tables are organized by category."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = VisualizationOrchestrator(Path(tmpdir))
            
            # Create mock table structure
            (Path(tmpdir) / "tables" / "weighting").mkdir(parents=True, exist_ok=True)
            (Path(tmpdir) / "tables" / "ranking").mkdir(parents=True, exist_ok=True)
            (Path(tmpdir) / "tables" / "ml").mkdir(parents=True, exist_ok=True)
            
            assert (Path(tmpdir) / "tables" / "weighting").exists()
            assert (Path(tmpdir) / "tables" / "ranking").exists()
            assert (Path(tmpdir) / "tables" / "ml").exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
