"""
tests/unit/test_temporal_stability.py
=====================================
Unit tests for temporal stability analysis module.

Test coverage:
- Window generation (correctness, boundary cases, edge cases)
- RMSD computation (mathematical correctness, symmetry properties)
- CV computation (mathematical correctness, NaN handling)
- Full temporal stability pipeline
"""

from __future__ import annotations

import math
import pytest
import numpy as np
import pandas as pd

from src.core.exceptions import FrameworkError
from src.core.schema import WeightVector
from src.mcdm.analysis.temporal_stability import (
    generate_windows,
    compute_rmsd,
    compute_cv_per_subcriteria,
    TemporalStabilityResult,
)


class TestWindowGeneration:
    """Tests for overlapping window generation."""

    def test_generate_windows_basic(self):
        """Test basic window generation with standard parameters."""
        years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        windows = generate_windows(years, window_size=5, n_windows=10)

        assert len(windows) == 10
        # First window: 2011-2015
        assert windows[0] == [2011, 2012, 2013, 2014, 2015]
        # Last window: 2020-2024
        assert windows[-1] == [2020, 2021, 2022, 2023, 2024]

        # All windows should have size 5
        for w in windows:
            assert len(w) == 5

    def test_generate_windows_maximum_possible(self):
        """Test generating maximum possible windows."""
        years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        windows = generate_windows(years, window_size=5, n_windows=None)

        # Max possible: 14 - 5 + 1 = 10
        assert len(windows) == 10

    def test_generate_windows_single_window(self):
        """Test generating single window."""
        years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        windows = generate_windows(years, window_size=5, n_windows=1)

        assert len(windows) == 1
        assert windows[0] == [2011, 2012, 2013, 2014, 2015]

    def test_generate_windows_full_span(self):
        """Test generating window that covers entire period."""
        years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        windows = generate_windows(years, window_size=14, n_windows=1)

        assert len(windows) == 1
        assert windows[0] == years

    def test_generate_windows_window_size_too_large(self):
        """Test error when window_size > available years."""
        years = [2011, 2012, 2013, 2014, 2015]
        with pytest.raises(FrameworkError):
            generate_windows(years, window_size=10)

    def test_generate_windows_n_windows_too_large(self):
        """Test error when n_windows > maximum possible."""
        years = [2011, 2012, 2013, 2014, 2015]
        with pytest.raises(FrameworkError):
            generate_windows(years, window_size=3, n_windows=10)

    def test_generate_windows_unsorted_years(self):
        """Test error when years are not sorted."""
        years = [2015, 2011, 2012, 2013, 2014]
        with pytest.raises(ValueError):
            generate_windows(years, window_size=3)

    def test_generate_windows_empty_list(self):
        """Test error with empty years list."""
        years = []
        with pytest.raises(ValueError):
            generate_windows(years, window_size=3)

    def test_generate_windows_single_year(self):
        """Test generating windows from single year."""
        years = [2011]
        windows = generate_windows(years, window_size=1, n_windows=1)
        assert windows == [[2011]]

    def test_generate_windows_invalid_window_size(self):
        """Test error with invalid window size."""
        years = [2011, 2012, 2013]
        with pytest.raises(ValueError):
            generate_windows(years, window_size=0)
        with pytest.raises(ValueError):
            generate_windows(years, window_size=-1)


class TestRMSDComputation:
    """Tests for RMSD between weight vectors."""

    def test_rmsd_identical_vectors(self):
        """RMSD of identical vectors should be zero."""
        w1 = WeightVector(
            labels=["SC11", "SC12", "SC13"],
            values=[0.4, 0.3, 0.3],
        )
        w2 = WeightVector(
            labels=["SC11", "SC12", "SC13"],
            values=[0.4, 0.3, 0.3],
        )
        rmsd = compute_rmsd(w1, w2)
        assert math.isclose(rmsd, 0.0, abs_tol=1e-9)

    def test_rmsd_symmetry(self):
        """RMSD should be symmetric: RMSD(w1, w2) == RMSD(w2, w1)."""
        w1 = WeightVector(
            labels=["SC11", "SC12", "SC13"],
            values=[0.5, 0.3, 0.2],
        )
        w2 = WeightVector(
            labels=["SC11", "SC12", "SC13"],
            values=[0.4, 0.3, 0.3],
        )
        rmsd_12 = compute_rmsd(w1, w2)
        rmsd_21 = compute_rmsd(w2, w1)
        assert math.isclose(rmsd_12, rmsd_21, abs_tol=1e-9)

    def test_rmsd_known_value(self):
        """Test RMSD against known analytical result."""
        w1 = WeightVector(
            labels=["A", "B", "C"],
            values=[0.5, 0.3, 0.2],
        )
        w2 = WeightVector(
            labels=["A", "B", "C"],
            values=[0.4, 0.3, 0.3],
        )
        # RMSD = sqrt(mean([(0.5-0.4)^2, (0.3-0.3)^2, (0.2-0.3)^2]))
        #      = sqrt(mean([0.01, 0, 0.01]))
        #      = sqrt(0.00667) ≈ 0.0816
        rmsd = compute_rmsd(w1, w2)
        expected = math.sqrt((0.01 + 0.0 + 0.01) / 3)
        assert math.isclose(rmsd, expected, abs_tol=1e-6)

    def test_rmsd_mismatched_lengths(self):
        """RMSD should error on mismatched vector lengths."""
        w1 = WeightVector(labels=["A", "B"], values=[0.5, 0.5])
        w2 = WeightVector(labels=["A", "B", "C"], values=[0.4, 0.3, 0.3])
        with pytest.raises(ValueError):
            compute_rmsd(w1, w2)

    def test_rmsd_large_difference(self):
        """Test RMSD with large weight differences."""
        w1 = WeightVector(labels=["A", "B", "C"], values=[1.0, 0.0, 0.0])
        w2 = WeightVector(labels=["A", "B", "C"], values=[0.0, 0.0, 1.0])
        rmsd = compute_rmsd(w1, w2)
        # RMSD = sqrt(mean([1, 0, 1])) = sqrt(2/3) ≈ 0.8165
        expected = math.sqrt(2.0 / 3.0)
        assert math.isclose(rmsd, expected, abs_tol=1e-6)


class TestCVComputation:
    """Tests for Coefficient of Variation computation."""

    def test_cv_identical_weights(self):
        """CV of identical weight vectors should be zero."""
        w1 = WeightVector(labels=["SC11", "SC12"], values=[0.5, 0.5])
        w2 = WeightVector(labels=["SC11", "SC12"], values=[0.5, 0.5])
        w3 = WeightVector(labels=["SC11", "SC12"], values=[0.5, 0.5])
        
        cv_dict = compute_cv_per_subcriteria([w1, w2, w3])
        assert math.isclose(cv_dict["SC11"], 0.0, abs_tol=1e-9)
        assert math.isclose(cv_dict["SC12"], 0.0, abs_tol=1e-9)

    def test_cv_known_value(self):
        """Test CV against known analytical result."""
        # Weights: [0.5, 0.3, 0.2], [0.4, 0.3, 0.3], [0.3, 0.3, 0.4]
        w1 = WeightVector(labels=["A", "B", "C"], values=[0.5, 0.3, 0.2])
        w2 = WeightVector(labels=["A", "B", "C"], values=[0.4, 0.3, 0.3])
        w3 = WeightVector(labels=["A", "B", "C"], values=[0.3, 0.3, 0.4])

        cv_dict = compute_cv_per_subcriteria([w1, w2, w3])

        # For "A": mean=0.4, std=sqrt(((0.5-0.4)^2 + (0.4-0.4)^2 + (0.3-0.4)^2)/2) = sqrt(0.02/2) = sqrt(0.01) = 0.1
        # CV_A = 0.1 / 0.4 = 0.25
        assert math.isclose(cv_dict["A"], 0.25, abs_tol=1e-6)

    def test_cv_empty_list(self):
        """CV computation should error on empty weight list."""
        with pytest.raises(ValueError):
            compute_cv_per_subcriteria([])

    def test_cv_zero_mean_weight(self):
        """CV should be NaN when mean weight is near zero."""
        w1 = WeightVector(labels=["A"], values=[0.0])
        w2 = WeightVector(labels=["A"], values=[0.0])
        
        cv_dict = compute_cv_per_subcriteria([w1, w2])
        assert np.isnan(cv_dict["A"])

    def test_cv_small_variance(self):
        """Test CV with small variance."""
        w1 = WeightVector(labels=["A"], values=[0.5])
        w2 = WeightVector(labels=["A"], values=[0.500001])
        
        cv_dict = compute_cv_per_subcriteria([w1, w2])
        assert cv_dict["A"] > 0.0  # Should be very small but positive


class TestTemporalStabilityResult:
    """Tests for TemporalStabilityResult dataclass."""

    def test_result_to_dict(self):
        """Test serialization to dictionary."""
        w1 = WeightVector(labels=["SC11", "SC12"], values=[0.5, 0.5])
        w2 = WeightVector(labels=["SC11", "SC12"], values=[0.4, 0.6])

        result = TemporalStabilityResult(
            window_years=[(2011, 2015), (2012, 2016)],
            window_weights=[w1, w2],
            rmsd_consecutive=[0.07],
            cv_per_subcriteria={"SC11": 0.1, "SC12": 0.15},
            cv_overall=0.125,
            rmsd_mean=0.07,
            rmsd_std=0.0,
        )

        result_dict = result.to_dict()
        assert len(result_dict["window_years"]) == 2
        assert result_dict["rmsd_mean"] == 0.07
        assert result_dict["cv_overall"] == 0.125

    def test_result_to_dataframe(self):
        """Test conversion to DataFrames."""
        w1 = WeightVector(labels=["SC11", "SC12"], values=[0.5, 0.5])
        w2 = WeightVector(labels=["SC11", "SC12"], values=[0.4, 0.6])

        result = TemporalStabilityResult(
            window_years=[(2011, 2015), (2012, 2016)],
            window_weights=[w1, w2],
            rmsd_consecutive=[0.07],
            cv_per_subcriteria={"SC11": 0.1, "SC12": 0.15},
            cv_overall=0.125,
            rmsd_mean=0.07,
            rmsd_std=0.0,
        )

        weights_df, metrics_df = result.to_dataframe()
        
        assert weights_df.shape == (2, 2)
        assert list(weights_df.columns) == ["SC11", "SC12"]
        assert weights_df.loc["W1_2011-2015", "SC11"] == 0.5
        assert weights_df.loc["W2_2012-2016", "SC11"] == 0.4


class TestTemporalStabilityIntegration:
    """Integration tests for full temporal stability workflow."""

    def test_temporal_stability_mock_workflow(self):
        """Test workflow with synthetic data (mocked weighting)."""
        # Create mock panel dict
        years = [2011, 2012, 2013, 2014, 2015, 2016]
        panel_dict = {
            year: pd.DataFrame({
                "Province": ["P01", "P02", "P03"],
                "SC11": [1.0, 1.5, 2.0],
                "SC12": [1.5, 2.0, 2.5],
                "SC13": [2.0, 2.5, 3.0],
            })
            for year in years
        }

        # Test window generation
        windows = generate_windows(list(panel_dict.keys()), window_size=3, n_windows=4)
        assert len(windows) == 4
        assert all(len(w) == 3 for w in windows)

    def test_rmsd_sequence_property(self):
        """Test that RMSD forms valid distance metric properties."""
        w1 = WeightVector(labels=["A", "B"], values=[0.5, 0.5])
        w2 = WeightVector(labels=["A", "B"], values=[0.6, 0.4])
        w3 = WeightVector(labels=["A", "B"], values=[0.7, 0.3])

        rmsd_12 = compute_rmsd(w1, w2)
        rmsd_23 = compute_rmsd(w2, w3)
        rmsd_13 = compute_rmsd(w1, w3)

        # Triangle inequality: d(1,3) <= d(1,2) + d(2,3)
        assert rmsd_13 <= rmsd_12 + rmsd_23 + 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
