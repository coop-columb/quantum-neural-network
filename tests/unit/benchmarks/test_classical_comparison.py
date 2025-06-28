"""
Tests for classical comparison utilities.
"""

import numpy as np
import pytest
import tensorflow as tf

from quantum_nn.benchmarks import run_classical_comparison
from quantum_nn.benchmarks.classical_comparison import (
    create_classical_model,
    wrap_classical_model,
)
from quantum_nn.models import QuantumModel


class TestClassicalComparison:
    """Test suite for classical comparison utilities."""

    def test_create_classical_model(self):
        """Test creating different classical models."""
        # Test MLP
        mlp = create_classical_model(
            model_type="mlp",
            input_shape=(10,),
            output_shape=(1,),
            task_type="classification",
        )

        assert isinstance(mlp, tf.keras.Model)
        assert mlp.output_shape == (None, 1)

        # Test CNN
        cnn = create_classical_model(
            model_type="cnn",
            input_shape=(28, 28, 1),
            output_shape=(10,),
            task_type="classification",
        )

        assert isinstance(cnn, tf.keras.Model)
        assert cnn.output_shape == (None, 10)

        # Test RNN
        rnn = create_classical_model(
            model_type="rnn",
            input_shape=(10, 5),
            output_shape=(1,),
            task_type="regression",
        )

        assert isinstance(rnn, tf.keras.Model)
        assert rnn.output_shape == (None, 1)

        # Test invalid model type
        with pytest.raises(ValueError):
            create_classical_model(
                model_type="invalid", input_shape=(10,), output_shape=(1,)
            )

    def test_wrap_classical_model(self):
        """Test wrapping a classical model."""
        # Create a simple keras model
        keras_model = tf.keras.Sequential(
            [tf.keras.layers.Dense(1, activation="sigmoid")]
        )

        keras_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Wrap it
        wrapped_model = wrap_classical_model(keras_model)

        # Check that it has QuantumModel methods
        assert hasattr(wrapped_model, "fit")
        assert hasattr(wrapped_model, "evaluate")
        assert hasattr(wrapped_model, "predict")
        assert hasattr(wrapped_model, "compile")
        assert hasattr(wrapped_model, "summary")

        # Check that it's using the original model under the hood
        assert wrapped_model.model is keras_model

    def test_run_classical_comparison(self):
        """Test running a classical comparison."""
        # Create a mock quantum model
        quantum_model = MockQuantumModel()

        # Create simple test data
        x_train = np.random.random((10, 5))
        y_train = np.random.randint(0, 2, (10, 1)).astype(np.float32)
        x_test = np.random.random((5, 5))
        y_test = np.random.randint(0, 2, (5, 1)).astype(np.float32)

        # Run comparison with MLP only
        results = run_classical_comparison(
            quantum_model=quantum_model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            task_name="test_task",
            classical_models=["mlp"],
            epochs=2,
            batch_size=5,
            output_dir=None,  # Skip saving to disk
        )

        # Check results
        assert isinstance(results, dict)
        assert "quantum_model" in results
        assert "classical_mlp" in results


class MockQuantumModel(QuantumModel):
    """Mock implementation of QuantumModel for testing."""

    def __init__(self):
        """Initialize mock model."""
        self.model = tf.keras.Sequential(
            [tf.keras.layers.Dense(1, activation="sigmoid")]
        )

        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def fit(self, x, y, **kwargs):
        """Mock fit method."""
        # Return a mock history object with random metrics
        history = {
            "loss": [0.5, 0.4],
            "accuracy": [0.6, 0.7],
            "val_loss": [0.55, 0.45],
            "val_accuracy": [0.55, 0.65],
        }
        return type("obj", (object,), {"history": history})

    def evaluate(self, x, y, **kwargs):
        """Mock evaluate method."""
        return [0.4, 0.7]

    def predict(self, x, **kwargs):
        """Mock predict method."""
        return np.random.random((len(x), 1))
