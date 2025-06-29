"""
Tests for quantum state visualization.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import numpy as np
import pennylane as qml
import pytest

from quantum_nn.visualization import (
    plot_state_amplitudes,
    plot_state_bloch,
    plot_state_city,
)


class TestStateVisualization:
    """Test suite for quantum state visualization."""

    def test_plot_state_amplitudes(self):
        """Test amplitude plot."""
        # Create a simple state
        state = np.array([0.7071, 0, 0, 0.7071j])  # Bell state

        # Test plotting
        fig = plot_state_amplitudes(state)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_state_bloch(self):
        """Test Bloch sphere plot."""
        # Create a simple state
        state = np.array([0.7071, 0, 0, 0.7071j])  # Bell state

        # Test plotting
        fig = plot_state_bloch(state)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_state_city(self):
        """Test city plot."""
        # Create a simple state
        state = np.zeros(16)  # 4-qubit state
        state[0] = 0.5
        state[5] = 0.5
        state[10] = 0.5
        state[15] = 0.5

        # Test plotting
        fig = plot_state_city(state)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)
