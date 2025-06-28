"""
Tests for the QuantumLayer class.
"""

import numpy as np
import pytest
import tensorflow as tf

from quantum_nn.circuits import ParameterizedCircuit
from quantum_nn.layers import QuantumLayer


class TestQuantumLayer:
    """Test suite for the QuantumLayer class."""

    def test_initialization(self):
        """Test that the layer initializes correctly."""
        circuit = ParameterizedCircuit(n_qubits=2, n_layers=1)
        layer = QuantumLayer(circuit)

        assert layer.n_parameters == 6  # 2 qubits * 1 layer * 3 rotations
        assert layer.input_scaling == "tanh"
        assert isinstance(layer.quantum_weights, tf.Variable)
        assert layer.quantum_weights.shape == (6,)

    def test_weight_initialization(self):
        """Test custom weight initialization."""
        circuit = ParameterizedCircuit(n_qubits=2, n_layers=1)
        initial_weights = np.zeros(6)
        layer = QuantumLayer(circuit, initial_weights=initial_weights)

        assert np.allclose(layer.quantum_weights.numpy(), np.zeros(6))

    def test_forward_pass(self):
        """Test forward pass through the layer."""
        circuit = ParameterizedCircuit(n_qubits=2, n_layers=1)
        initial_weights = np.zeros(6)
        layer = QuantumLayer(circuit, initial_weights=initial_weights)

        # Create a batch of 3 inputs, each with 2 features
        inputs = tf.constant(
            [
                [0.0, 0.0],  # Should give [1, 1]
                [np.pi, 0.0],  # Should give [-1, 1] after tanh scaling
                [0.0, np.pi],  # Should give [1, -1] after tanh scaling
            ],
            dtype=tf.float32,
        )

        outputs = layer(inputs)

        # Check output shape: 3 examples, 2 output features
        assert outputs.shape == (3, 2)

        # First example: no rotation, should be [1, 1]
        assert np.allclose(outputs[0], [1, 1])

    def test_input_scaling(self):
        """Test different input scaling methods."""
        circuit = ParameterizedCircuit(n_qubits=2, n_layers=1)
        initial_weights = np.zeros(6)

        # Test tanh scaling
        layer_tanh = QuantumLayer(
            circuit, initial_weights=initial_weights, input_scaling="tanh"
        )

        # Test sigmoid scaling
        layer_sigmoid = QuantumLayer(
            circuit, initial_weights=initial_weights, input_scaling="sigmoid"
        )

        # Test no scaling
        layer_none = QuantumLayer(
            circuit, initial_weights=initial_weights, input_scaling=None
        )

        # Create a single input with large values
        inputs = tf.constant([[10.0, -10.0]], dtype=tf.float32)

        # Outputs with tanh scaling should be bounded to [-1, 1]
        outputs_tanh = layer_tanh._scale_inputs(inputs)
        assert np.all(outputs_tanh <= 1.0) and np.all(outputs_tanh >= -1.0)

        # Outputs with sigmoid scaling should be bounded to [0, 1]
        outputs_sigmoid = layer_sigmoid._scale_inputs(inputs)
        assert np.all(outputs_sigmoid <= 1.0) and np.all(outputs_sigmoid >= 0.0)

        # Outputs with no scaling should be unchanged
        outputs_none = layer_none._scale_inputs(inputs)
        assert np.allclose(outputs_none, inputs)
