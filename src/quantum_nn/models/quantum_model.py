import tensorflow as tf
from typing import List, Optional, Union, Dict, Any

from ..layers.quantum_layer import QuantumLayer


class QuantumModel(tf.keras.Model):
    """
    A quantum neural network model that integrates quantum layers with classical neural network
    components. This model provides a high-level interface for building hybrid quantum-classical
    models.

    Attributes:
        quantum_layers (List[QuantumLayer]): List of quantum layers in the model
        classical_pre_layers (List[tf.keras.layers.Layer]): Classical layers before quantum processing
        classical_post_layers (List[tf.keras.layers.Layer]): Classical layers after quantum processing
        name (str): Name of the model
    """

    def __init__(
        self,
        quantum_layers: List[QuantumLayer],
        classical_pre_layers: Optional[List[tf.keras.layers.Layer]] = None,
        classical_post_layers: Optional[List[tf.keras.layers.Layer]] = None,
        name: str = "quantum_model",
        **kwargs,
    ):
        """
        Initialize the quantum model.

        Args:
            quantum_layers: List of quantum layers to include in the model
            classical_pre_layers: Classical layers to apply before quantum processing
            classical_post_layers: Classical layers to apply after quantum processing
            name: Name of the model
        """
        super().__init__(name=name, **kwargs)

        self.quantum_layers = quantum_layers
        self.classical_pre_layers = classical_pre_layers or []
        self.classical_post_layers = classical_post_layers or []

        # Add output layer if not provided
        if not self.classical_post_layers or not isinstance(
            self.classical_post_layers[-1], tf.keras.layers.Dense
        ):
            # Add a default output layer
            self.classical_post_layers.append(tf.keras.layers.Dense(1))

    def call(self, inputs, training=None):
        """
        Forward pass of the quantum model.

        Args:
            inputs: Input tensor
            training: Whether the model is being called during training

        Returns:
            Model output tensor
        """
        x = inputs

        # Apply classical pre-processing layers
        for layer in self.classical_pre_layers:
            x = layer(x, training=training)

        # Apply quantum layers and concatenate their outputs
        quantum_outputs = []
        for q_layer in self.quantum_layers:
            q_out = q_layer(x, training=training)
            quantum_outputs.append(q_out)

        # Concatenate quantum outputs if there are multiple quantum layers
        if len(quantum_outputs) > 1:
            x = tf.concat(quantum_outputs, axis=-1)
        else:
            x = quantum_outputs[0]

        # Apply classical post-processing layers
        for layer in self.classical_post_layers:
            x = layer(x, training=training)

        return x

    def build_graph(self, input_shape):
        """
        Build the model graph.

        Args:
            input_shape: Shape of the input tensor

        Returns:
            Model instance with built graph
        """
        inputs = tf.keras.Input(shape=input_shape[1:])
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def summary(self, line_length=None, positions=None, print_fn=None):
        """
        Print a summary of the model architecture.

        Args:
            line_length: Total length of printed lines
            positions: Relative or absolute positions of log elements in each line
            print_fn: Print function to use
        """
        # Build a functional model for proper summary
        if not self.built:
            self.build(self.input_shape)

        # Count number of quantum and classical parameters
        n_quantum_params = sum(layer.n_parameters for layer in self.quantum_layers)
        n_classical_params = sum(
            tf.size(var).numpy()
            for var in self.trainable_variables
            if "quantum_parameters" not in var.name
        )

        # Print standard summary
        super().summary(line_length, positions, print_fn)

        # Print additional quantum-specific information
        print_fn = print_fn or print
        print_fn("\nQuantum Model Information:")
        print_fn(f"  Number of quantum layers: {len(self.quantum_layers)}")
        print_fn(f"  Number of quantum parameters: {n_quantum_params}")
        print_fn(f"  Number of classical parameters: {n_classical_params}")
        print_fn(f"  Total parameters: {n_quantum_params + n_classical_params}")

    def get_config(self):
        """Get model configuration for serialization."""
        config = super().get_config()
        # Note: Full serialization would require custom logic to handle quantum layers
        return config
