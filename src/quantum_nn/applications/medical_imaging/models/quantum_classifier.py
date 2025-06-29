"""
Quantum classifier for medical imaging tasks.

This module implements quantum neural networks specifically designed
for medical image classification tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from quantum_nn.circuits import ParameterizedCircuit
from quantum_nn.layers import QuantumLayer
from quantum_nn.models import QuantumModel

logger = logging.getLogger(__name__)


class MedicalQuantumClassifier(tf.keras.Model):
    """
    Quantum classifier optimized for medical imaging tasks.

    This model uses quantum neural networks to classify medical images,
    with special consideration for small datasets and interpretability.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_qubits: int = 8,
        n_layers: int = 3,
        circuit_type: str = "strongly_entangling",
        encoding_type: str = "angle",
        measurement_type: str = "expectation",
        n_classes: int = 2,
        quantum_device: str = "default.qubit",
        **kwargs,
    ):
        """
        Initialize medical quantum classifier.

        Args:
            input_shape: Shape of input features
            n_qubits: Number of qubits
            n_layers: Number of quantum layers (stored for reference)
            circuit_type: Type of quantum circuit (stored for reference)
            encoding_type: Data encoding method (stored for reference)
            measurement_type: Measurement strategy
            n_classes: Number of output classes
            quantum_device: Quantum simulation backend (stored for reference)
            **kwargs: Additional arguments
        """
        # Initialize parent class first
        super().__init__(**kwargs)

        self.input_shape_custom = input_shape
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_type = circuit_type
        self.encoding_type = encoding_type
        self.measurement_type = measurement_type
        self.n_classes = n_classes
        self.quantum_device = quantum_device

        # Build the model layers
        self._build_layers()

        # Initialize medical-specific components
        self._initialize_medical_metrics()

        logger.info(
            f"Initialized MedicalQuantumClassifier with {n_qubits} qubits, "
            f"{n_layers} layers, {circuit_type} circuit"
        )

    def _build_layers(self):
        """Build the quantum classifier layers."""
        # Feature processing layers
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.2)

        # Quantum layer - use only the parameters that QuantumLayer accepts
        self.quantum_layer = QuantumLayer(
            n_qubits=self.n_qubits, measurement_type=self.measurement_type
        )

        # Post-quantum processing
        self.dense2 = tf.keras.layers.Dense(32, activation="relu")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.3)

        # Output layer
        if self.n_classes == 2:
            self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")
        else:
            self.output_layer = tf.keras.layers.Dense(
                self.n_classes, activation="softmax"
            )

    def call(self, inputs, training=None):
        """Forward pass of the model."""
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.quantum_layer(x)
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

    def _initialize_medical_metrics(self):
        """Initialize medical-specific metrics."""
        self.medical_metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]

    def compile_for_medical_imaging(
        self,
        learning_rate: float = 0.001,
        loss: Optional[str] = None,
        optimizer: Optional[str] = None,
    ):
        """
        Compile model with medical imaging specific settings.

        Args:
            learning_rate: Learning rate for optimizer
            loss: Loss function (defaults based on n_classes)
            optimizer: Optimizer name
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if loss is None:
            loss = (
                "binary_crossentropy"
                if self.n_classes == 2
                else "categorical_crossentropy"
            )

        self.compile(optimizer=optimizer, loss=loss, metrics=self.medical_metrics)

        logger.info("Model compiled for medical imaging")

    def fit_with_medical_callbacks(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 50,
        batch_size: int = 16,
        class_weight: Optional[Dict[int, float]] = None,
        **kwargs,
    ):
        """
        Train model with medical-specific callbacks.

        Args:
            x_train: Training data
            y_train: Training labels
            validation_data: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            class_weight: Class weights for imbalanced data
            **kwargs: Additional fit arguments

        Returns:
            Training history
        """
        # Medical-specific callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc" if validation_data else "auc",
                patience=10,
                restore_best_weights=True,
                mode="max",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if validation_data else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "medical_quantum_best.h5",
                monitor="val_auc" if validation_data else "auc",
                save_best_only=True,
                mode="max",
            ),
        ]

        # Add any user callbacks
        if "callbacks" in kwargs:
            callbacks.extend(kwargs["callbacks"])

        kwargs["callbacks"] = callbacks

        # Train
        history = self.fit(
            x_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            **kwargs,
        )

        return history

    def predict_with_uncertainty(
        self, x: np.ndarray, n_iterations: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation.

        Args:
            x: Input data
            n_iterations: Number of forward passes

        Returns:
            Tuple of (predictions, uncertainties)
        """
        # Enable dropout during prediction
        predictions = []

        for _ in range(n_iterations):
            # Make prediction with dropout enabled
            pred = self(x, training=True)
            predictions.append(pred.numpy())

        predictions = np.array(predictions)

        # Calculate mean and std
        mean_predictions = np.mean(predictions, axis=0)
        uncertainties = np.std(predictions, axis=0)

        return mean_predictions, uncertainties

    def explain_prediction(
        self, x_sample: np.ndarray, method: str = "gradient"
    ) -> np.ndarray:
        """
        Explain model predictions using various methods.

        Args:
            x_sample: Single input sample
            method: Explanation method ('gradient' or 'permutation')

        Returns:
            Feature importance scores
        """
        if method == "gradient":
            # Use gradients to measure feature importance
            x_sample = tf.convert_to_tensor(x_sample)
            with tf.GradientTape() as tape:
                tape.watch(x_sample)
                prediction = self(x_sample)

            gradients = tape.gradient(prediction, x_sample)
            importance = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()

        elif method == "permutation":
            # Use permutation importance
            baseline_pred = self(x_sample).numpy()
            importance = np.zeros(x_sample.shape[1])

            for i in range(x_sample.shape[1]):
                # Permute feature i
                x_permuted = x_sample.numpy().copy()
                np.random.shuffle(x_permuted[:, i])

                # Calculate change in prediction
                permuted_pred = self(tf.constant(x_permuted)).numpy()
                importance[i] = np.mean(np.abs(baseline_pred - permuted_pred))

        else:
            raise ValueError(f"Unknown importance method: {method}")

        return importance

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape_custom,
                "n_qubits": self.n_qubits,
                "n_layers": self.n_layers,
                "circuit_type": self.circuit_type,
                "encoding_type": self.encoding_type,
                "measurement_type": self.measurement_type,
                "n_classes": self.n_classes,
                "quantum_device": self.quantum_device,
            }
        )
        return config


def create_medical_quantum_classifier(
    input_shape: Tuple[int, ...],
    n_classes: int = 2,
    n_qubits: Optional[int] = None,
    circuit_complexity: str = "medium",
    **kwargs,
) -> MedicalQuantumClassifier:
    """
    Factory function to create a medical quantum classifier.

    Args:
        input_shape: Shape of input data
        n_classes: Number of output classes
        n_qubits: Number of qubits (auto-calculated if None)
        circuit_complexity: Complexity level ('simple', 'medium', 'complex')
        **kwargs: Additional arguments

    Returns:
        Configured MedicalQuantumClassifier
    """
    # Auto-calculate number of qubits based on input dimension
    input_dim = np.prod(input_shape)

    if n_qubits is None:
        # Use log scale with minimum of 4 qubits
        n_qubits = max(4, min(12, int(np.log2(input_dim)) + 2))

    # Set circuit parameters based on complexity
    complexity_configs = {
        "simple": {"n_layers": 1, "circuit_type": "strongly_entangling"},
        "medium": {"n_layers": 3, "circuit_type": "strongly_entangling"},
        "complex": {"n_layers": 5, "circuit_type": "convolution"},
    }

    config = complexity_configs.get(circuit_complexity, complexity_configs["medium"])
    config.update(kwargs)

    # Create and return the classifier
    classifier = MedicalQuantumClassifier(
        input_shape=input_shape, n_qubits=n_qubits, n_classes=n_classes, **config
    )

    logger.info(
        f"Created medical quantum classifier: {n_qubits} qubits, "
        f"{config['n_layers']} layers, {circuit_complexity} complexity"
    )

    return classifier
