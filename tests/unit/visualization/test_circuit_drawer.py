"""
Tests for circuit visualization.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import numpy as np
import pytest
import pennylane as qml

from quantum_nn.circuits import ParameterizedCircuit
from quantum_nn.visualization import draw_circuit


class TestCircuitDrawer:
    """Test suite for circuit drawer."""

    def test_draw_circuit_matplotlib(self):
        """Test drawing circuit with matplotlib output."""
        # Create a simple circuit
        circuit = ParameterizedCircuit(n_qubits=2, template="strongly_entangling")
        params = np.zeros(circuit.get_n_params())

        # Test with matplotlib output
        fig = draw_circuit(circuit, params, output_format="matplotlib")

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_draw_circuit_text(self):
        """Test drawing circuit with text output."""
        # Create a simple circuit
        circuit = ParameterizedCircuit(n_qubits=2, template="strongly_entangling")
        params = np.zeros(circuit.get_n_params())

        # Test with text output
        text = draw_circuit(circuit, params, output_format="text")

        assert text is not None
        assert isinstance(text, str)
        assert len(text) > 0

    def test_draw_circuit_invalid_format(self):
        """Test error handling for invalid output format."""
        # Create a simple circuit
        circuit = ParameterizedCircuit(n_qubits=2, template="strongly_entangling")
        params = np.zeros(circuit.get_n_params())

        # Test with invalid output format
        with pytest.raises(ValueError):
            draw_circuit(circuit, params, output_format="invalid")
