"""
Tests for state visualization.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import numpy as np
import pytest

from quantum_nn.visualization import (
    plot_state_amplitudes,
    plot_state_bloch,
    plot_state_city,
)


class TestStateVisualization:
    """Test suite for state visualization."""

    def test_plot_state_amplitudes(self):
        """Test amplitude plotting function."""
        # Create a simple 2-qubit state |00⟩ + |11⟩
        state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])

        # Test basic functionality
        fig = plot_state_amplitudes(state)
        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

        # Test with phase disabled
        fig_no_phase = plot_state_amplitudes(state, show_phase=False)
        assert fig_no_phase is not None

        # Test with custom labels
        fig_custom = plot_state_amplitudes(
            state, labels=["00", "01", "10", "11"]
        )
        assert fig_custom is not None

        # Test error handling for invalid state size
        with pytest.raises(ValueError):
            invalid_state = np.array([1, 0, 0])  # Not a power of 2
            plot_state_amplitudes(invalid_state)

    def test_plot_state_bloch(self):
        """Test Bloch sphere plotting function."""
        # Create a simple 2-qubit state
        state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])

        # Test basic functionality
        fig = plot_state_bloch(state)
        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

        # Test with specific qubit indices
        fig_specific = plot_state_bloch(state, qubit_indices=[0])
        assert fig_specific is not None

        # Test with single qubit state
        single_qubit_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        fig_single = plot_state_bloch(single_qubit_state)
        assert fig_single is not None

        # Test error handling for invalid state size
        with pytest.raises(ValueError):
            invalid_state = np.array([1, 0, 0])  # Not a power of 2
            plot_state_bloch(invalid_state)

        # Test error handling for invalid qubit indices
        with pytest.raises(ValueError):
            plot_state_bloch(state, qubit_indices=[5])  # Out of range

    def test_plot_state_city(self):
        """Test city plot function."""
        # Create a simple 2-qubit state
        state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])

        # Test basic functionality
        fig = plot_state_city(state)
        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

        # Test with different colormap
        fig_plasma = plot_state_city(state, colormap="plasma")
        assert fig_plasma is not None

        # Test without wireframe
        fig_no_wire = plot_state_city(state, show_wireframe=False)
        assert fig_no_wire is not None

        # Test with 3-qubit state (odd number of qubits)
        state_3q = np.zeros(8)
        state_3q[0] = state_3q[7] = 1 / np.sqrt(2)  # |000⟩ + |111⟩
        fig_3q = plot_state_city(state_3q)
        assert fig_3q is not None

        # Test error handling for invalid state size
        with pytest.raises(ValueError):
            invalid_state = np.array([1, 0, 0])  # Not a power of 2
            plot_state_city(invalid_state)

    def test_amplitude_threshold_handling(self):
        """Test that small amplitudes are handled correctly."""
        # Create state with very small amplitudes
        state = np.array([1.0, 1e-8, 1e-8, 1e-8])
        state = state / np.linalg.norm(state)

        # Should work without errors
        fig = plot_state_city(state)
        assert fig is not None

    def test_normalization_independence(self):
        """Test that functions work with unnormalized states."""
        # Create unnormalized state (functions should handle this gracefully)
        state = np.array([2.0, 0, 0, 2.0])  # Not normalized

        # All functions should work
        fig1 = plot_state_amplitudes(state)
        fig2 = plot_state_bloch(state)
        fig3 = plot_state_city(state)

        assert all(fig is not None for fig in [fig1, fig2, fig3])