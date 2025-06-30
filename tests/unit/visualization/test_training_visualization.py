"""
Tests for training visualization.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import numpy as np
import pytest

from quantum_nn.visualization import plot_loss_landscape, plot_training_trajectory


class TestTrainingVisualization:
    """Test suite for training visualization."""

    def test_plot_loss_landscape(self):
        """Test loss landscape plot."""

        # Define a simple loss function
        def loss_fn(params):
            return np.sum((params - 1.0) ** 2)

        # Create parameters
        params = np.zeros(5)

        # Test plotting
        fig = plot_loss_landscape(loss_fn, params)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

        # Test with different plot types
        fig_contour = plot_loss_landscape(loss_fn, params, plot_type="contour")
        assert fig_contour is not None

        fig_both = plot_loss_landscape(loss_fn, params, plot_type="both")
        assert fig_both is not None

    def test_plot_training_trajectory(self):
        """Test training trajectory plot."""
        # Create a sample history dictionary
        history = {
            "loss": [0.5, 0.4, 0.3, 0.2],
            "accuracy": [0.6, 0.7, 0.8, 0.9],
            "val_loss": [0.55, 0.45, 0.35, 0.25],
            "val_accuracy": [0.55, 0.65, 0.75, 0.85],
        }

        # Test plotting all metrics
        fig = plot_training_trajectory(history)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

        # Test plotting specific metrics
        fig_specific = plot_training_trajectory(history, metrics=["loss"])
        assert fig_specific is not None

        # Test without validation
        fig_no_val = plot_training_trajectory(history, include_validation=False)
        assert fig_no_val is not None
