"""
Tests for the ParameterizedCircuit class.
"""
import numpy as np
import pytest

from quantum_nn.circuits import ParameterizedCircuit


class TestParameterizedCircuit:
    """Test suite for the ParameterizedCircuit class."""

    def test_initialization(self):
        """Test that the circuit initializes correctly."""
        circuit = ParameterizedCircuit(n_qubits=3, n_layers=2)
        assert circuit.n_qubits == 3
        assert circuit.n_layers == 2
        assert circuit.entanglement == "linear"
        assert circuit.n_params == 18  # 3 qubits * 2 layers * 3 rotations

    def test_parameter_count(self):
        """Test parameter counting logic."""
        circuits = [
            (2, 1, 6),   # 2 qubits, 1 layer = 6 params
            (3, 2, 18),  # 3 qubits, 2 layers = 18 params
            (4, 3, 36),  # 4 qubits, 3 layers = 36 params
        ]
        
        for n_qubits, n_layers, expected_params in circuits:
            circuit = ParameterizedCircuit(n_qubits=n_qubits, n_layers=n_layers)
            assert circuit.get_n_params() == expected_params

    def test_circuit_execution(self):
        """Test circuit execution with parameters."""
        circuit = ParameterizedCircuit(n_qubits=2, n_layers=1)
        params = np.zeros(6)  # 2 qubits * 1 layer * 3 rotations
        
        # Execute circuit with zero parameters
        result = circuit(params)
        
        # With zero rotations and Pauli-Z measurements, expect [1, 1]
        assert result.shape == (2,)
        assert np.allclose(result, [1, 1])
        
    def test_data_encoding(self):
        """Test data encoding in the circuit."""
        circuit = ParameterizedCircuit(n_qubits=2, n_layers=1)
        params = np.zeros(6)
        inputs = np.array([np.pi, 0])  # Flip first qubit with RX(pi)
        
        result = circuit(params, inputs)
        
        # First qubit should be flipped, second unchanged
        assert np.allclose(result, [-1, 1])
