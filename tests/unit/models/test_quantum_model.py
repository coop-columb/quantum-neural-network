"""
Tests for the QuantumModel class.
"""

import numpy as np
import pytest
import tensorflow as tf

from quantum_nn.circuits import ParameterizedCircuit
from quantum_nn.layers import QuantumLayer
from quantum_nn.models import QuantumModel


class TestQuantumModel:
    """Test suite for the QuantumModel class."""

    def test_initialization_with_config(self):
        """Test model initialization with configuration dictionary."""
        model = QuantumModel(
            [
                {"type": "quantum", "n_qubits": 2, "n_layers": 1},
                {"type": "dense", "units": 10, "activation": "relu"},
                {"type": "dense", "units": 1, "activation": "sigmoid"},
            ]
        )

        # Check that model was created successfully
        assert len(model.model.layers) == 3
        assert len(model.quantum_layers) == 1

    def test_initialization_with_instances(self):
        """Test model initialization with layer instances."""
        circuit = ParameterizedCircuit(n_qubits=2, n_layers=1)
        quantum_layer = QuantumLayer(circuit=circuit)

        model = QuantumModel(
            [
                quantum_layer,
                {"type": "dense", "units": 1, "activation": "sigmoid"},
            ]
        )

        # Check that model was created successfully
        assert len(model.model.layers) == 2
        assert len(model.quantum_layers) == 1
        assert model.quantum_layers[0] is quantum_layer

    def test_compile_and_fit(self):
        """Test model compilation and basic fitting."""
        # Create a simple model
        model = QuantumModel(
            [
                {"type": "quantum", "n_qubits": 2, "n_layers": 1},
                {"type": "dense", "units": 1, "activation": "sigmoid"},
            ]
        )

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Create small synthetic dataset
        x = np.random.random((10, 2))
        y = np.random.randint(0, 2, (10, 1)).astype(np.float32)

        # Test fitting for 1 epoch (just check that it runs)
        history = model.fit(x, y, epochs=1, verbose=0)

        # Check that history contains expected metrics
        assert "loss" in history.history
        assert "accuracy" in history.history

    def test_prediction(self):
        """Test model prediction."""
        # Create a simple model with fixed weights for deterministic output
        circuit = ParameterizedCircuit(n_qubits=2, n_layers=1)
        quantum_layer = QuantumLayer(
            circuit=circuit,
            initial_weights=np.zeros(6),  # Zero weights for predictable output
        )

        model = QuantumModel(
            [
                quantum_layer,
                {"type": "dense", "units": 1, "activation": "sigmoid"},
            ]
        )

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy")

        # Create simple input - a single example with two features
        x = np.array([[0.0, 0.0]])

        # Get predictions
        predictions = model.predict(x)

        # Check prediction shape
        assert predictions.shape == (1, 1)
