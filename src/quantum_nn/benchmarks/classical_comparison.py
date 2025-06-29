"""
Classical model comparison for quantum neural networks.

This module provides utilities for comparing quantum neural networks
with classical machine learning models on the same tasks.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from quantum_nn.benchmarks.benchmark_runner import BenchmarkRunner
from quantum_nn.models import QuantumModel


def create_classical_model(
    model_type: str,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    task_type: str = "classification",
    **kwargs,
) -> tf.keras.Model:
    """
    Create a classical model for comparison.

    Args:
        model_type: Type of model ('mlp', 'cnn', 'rnn')
        input_shape: Shape of input data
        output_shape: Shape of output data
        task_type: Type of task ('classification' or 'regression')
        **kwargs: Additional model parameters

    Returns:
        Classical TensorFlow model
    """
    # Determine output activation and loss based on task type
    if task_type == "classification":
        output_dim = output_shape[0] if len(output_shape) > 0 else 1
        output_activation = "sigmoid" if output_dim == 1 else "softmax"
        loss = "binary_crossentropy" if output_dim == 1 else "categorical_crossentropy"
    else:  # regression
        output_dim = output_shape[0] if len(output_shape) > 0 else 1
        output_activation = "linear"
        loss = "mse"

    # Create model based on type
    if model_type == "mlp":
        # Multilayer Perceptron
        n_layers = kwargs.get("n_layers", 2)
        hidden_dim = kwargs.get("hidden_dim", 64)
        dropout_rate = kwargs.get("dropout_rate", 0.2)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        model.add(tf.keras.layers.Flatten())

        for _ in range(n_layers):
            model.add(tf.keras.layers.Dense(hidden_dim, activation="relu"))
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(output_dim, activation=output_activation))

    elif model_type == "cnn":
        # Convolutional Neural Network
        # Assume 2D input for images
        if len(input_shape) < 2:
            raise ValueError("CNN requires at least 2D input")

        n_conv_layers = kwargs.get("n_conv_layers", 2)
        filters = kwargs.get("filters", [32, 64])
        kernel_size = kwargs.get("kernel_size", (3, 3))
        pool_size = kwargs.get("pool_size", (2, 2))
        dense_dim = kwargs.get("dense_dim", 128)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))

        # Add convolutional layers
        for i in range(n_conv_layers):
            f = filters[i] if i < len(filters) else filters[-1]
            model.add(tf.keras.layers.Conv2D(f, kernel_size, activation="relu"))
            model.add(tf.keras.layers.MaxPooling2D(pool_size))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(dense_dim, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(output_dim, activation=output_activation))

    elif model_type == "rnn":
        # Recurrent Neural Network
        # Assume sequential input
        if len(input_shape) < 1:
            raise ValueError("RNN requires at least 1D input")

        rnn_type = kwargs.get("rnn_type", "lstm")
        rnn_units = kwargs.get("rnn_units", 64)
        bidirectional = kwargs.get("bidirectional", False)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))

        # Create RNN layer based on type
        if rnn_type.lower() == "lstm":
            rnn_layer = tf.keras.layers.LSTM(rnn_units, return_sequences=False)
        elif rnn_type.lower() == "gru":
            rnn_layer = tf.keras.layers.GRU(rnn_units, return_sequences=False)
        else:
            rnn_layer = tf.keras.layers.SimpleRNN(rnn_units, return_sequences=False)

        # Wrap in bidirectional if requested
        if bidirectional:
            rnn_layer = tf.keras.layers.Bidirectional(rnn_layer)

        model.add(rnn_layer)
        model.add(tf.keras.layers.Dense(output_dim, activation=output_activation))

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Compile model
    model.compile(
        optimizer=kwargs.get("optimizer", "adam"),
        loss=loss,
        metrics=kwargs.get(
            "metrics", ["accuracy"] if task_type == "classification" else ["mae"]
        ),
    )

    return model


def wrap_classical_model(model: tf.keras.Model) -> QuantumModel:
    """
    Wrap a classical model in the QuantumModel interface for comparison.

    Args:
        model: Classical TensorFlow model

    Returns:
        QuantumModel-compatible wrapper
    """

    class ClassicalModelWrapper(QuantumModel):
        def __init__(self, keras_model):
            self.model = keras_model

        def compile(self, **kwargs):
            self.model.compile(**kwargs)

        def fit(self, x, y, **kwargs):
            return self.model.fit(x, y, **kwargs)

        def evaluate(self, x, y, **kwargs):
            return self.model.evaluate(x, y, **kwargs)

        def predict(self, x, **kwargs):
            return self.model.predict(x, **kwargs)

        def summary(self):
            return self.model.summary()

    return ClassicalModelWrapper(model)


def run_classical_comparison(
    quantum_model: QuantumModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task_name: str,
    quantum_model_name: str = "quantum_model",
    classical_models: Optional[List[str]] = None,
    task_type: str = "classification",
    epochs: int = 10,
    batch_size: int = 16,
    output_dir: str = "./experiments/results",
) -> Dict[str, Dict[str, Any]]:
    """
    Run a comparison between quantum and classical models.

    Args:
        quantum_model: Quantum neural network model
        x_train: Training input data
        y_train: Training target data
        x_test: Test input data
        y_test: Test target data
        task_name: Name of the benchmark task
        quantum_model_name: Name of the quantum model
        classical_models: List of classical model types to compare with
        task_type: Type of task ('classification' or 'regression')
        epochs: Number of training epochs
        batch_size: Batch size for training
        output_dir: Directory to save results

    Returns:
        Dictionary mapping model names to benchmark results
    """
    # Default classical models if not specified
    if classical_models is None:
        classical_models = ["mlp", "cnn"] if len(x_train.shape) > 2 else ["mlp"]

    # Create benchmark runner
    benchmark_runner = BenchmarkRunner(output_dir=output_dir)

    # Create dictionary of models to compare
    models = {quantum_model_name: quantum_model}

    # Add classical models
    for model_type in classical_models:
        try:
            # Create classical model
            classical_model = create_classical_model(
                model_type=model_type,
                input_shape=x_train.shape[1:],
                output_shape=y_train.shape[1:] if y_train.ndim > 1 else (1,),
                task_type=task_type,
            )

            # Wrap in QuantumModel interface
            models[f"classical_{model_type}"] = wrap_classical_model(classical_model)
        except Exception as e:
            print(f"Error creating {model_type} model: {e}")

    # Run comparison
    results = benchmark_runner.compare_models(
        models=models,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        task_name=task_name,
        epochs=epochs,
        batch_size=batch_size,
    )

    return results
