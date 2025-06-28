import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any

from ..circuits.parameterized_circuit import ParameterizedCircuit


class QuantumLayer(tf.keras.layers.Layer):
    """
    A quantum layer implementation that integrates with TensorFlow's Keras API.

    This layer wraps a parameterized quantum circuit and handles the conversion
    between classical data and quantum states, as well as the measurement of
    quantum states to produce classical outputs.

    Attributes:
        circuit (ParameterizedCircuit): The parameterized quantum circuit to execute
        measurement_indices (List[int]): Indices of qubits to measure
        readout_op (Optional[Any]): Custom readout operation, if None defaults to Z measurements
        expectation (bool): Whether to compute expectation values (True) or samples (False)
        n_samples (int): Number of measurement samples when expectation is False
        trainable (bool): Whether the quantum circuit parameters are trainable
        name (str): Name of the layer
    """

    def __init__(
        self,
        circuit: ParameterizedCircuit,
        measurement_indices: Optional[List[int]] = None,
        readout_op: Optional[Any] = None,
        expectation: bool = True,
        n_samples: int = 1000,
        trainable: bool = True,
        name: str = "quantum_layer",
        **kwargs,
    ):
        """
        Initialize the quantum layer.

        Args:
            circuit: The parameterized quantum circuit to execute
            measurement_indices: Indices of qubits to measure, defaults to all qubits
            readout_op: Custom readout operation, defaults to Z measurements
            expectation: Whether to compute expectation values (True) or samples (False)
            n_samples: Number of measurement samples when expectation is False
            trainable: Whether the quantum circuit parameters are trainable
            name: Name of the layer
        """
        super().__init__(name=name, trainable=trainable, **kwargs)

        self.circuit = circuit
        self.n_qubits = circuit.n_qubits

        # Default to measuring all qubits if not specified
        self.measurement_indices = measurement_indices or list(range(self.n_qubits))

        self.expectation = expectation
        self.n_samples = n_samples

        # Set up the quantum circuit
        self.quantum_circuit = circuit.circuit

        # Set up the trainable parameters
        self.n_parameters = circuit.n_parameters
        param_shape = (self.n_parameters,)
        self.params = self.add_weight(
            name="quantum_parameters",
            shape=param_shape,
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
            trainable=trainable,
        )

        # Set up the measurement operators
        self.readout_op = readout_op
        if self.readout_op is None:
            qubits = circuit.qubits
            self.readout_op = [cirq.Z(qubits[i]) for i in self.measurement_indices]

    def build(self, input_shape):
        """Build the layer."""
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Forward pass of the quantum layer.

        Args:
            inputs: Input tensor to encode into the quantum circuit
            training: Whether the layer is being called during training

        Returns:
            Tensor containing measurement results from the quantum circuit
        """
        # Convert inputs to the appropriate format for the circuit
        batch_size = tf.shape(inputs)[0]

        # Create parameterized circuits for each input in the batch
        circuits = self._prepare_circuits(inputs)

        # Perform the quantum computation
        if self.expectation:
            # Use expectation values
            expectation_layer = tfq.layers.Expectation()
            outputs = expectation_layer(
                circuits,
                operators=self.readout_op,
                symbol_names=self.circuit.parameter_names,
                symbol_values=tf.tile(tf.expand_dims(self.params, 0), [batch_size, 1]),
            )
        else:
            # Use sampling
            sample_layer = tfq.layers.Sample()
            outputs = sample_layer(
                circuits,
                operators=self.readout_op,
                repetitions=self.n_samples,
                symbol_names=self.circuit.parameter_names,
                symbol_values=tf.tile(tf.expand_dims(self.params, 0), [batch_size, 1]),
            )
            # Convert samples to probabilities
            outputs = tf.reduce_mean(tf.cast(outputs, tf.float32), axis=-1)

        return outputs

    def _prepare_circuits(self, inputs):
        """
        Prepare quantum circuits with inputs encoded.

        Args:
            inputs: Input tensor to encode into the quantum circuit

        Returns:
            TensorFlow Quantum circuits with inputs encoded
        """
        # Generate circuits with data encoded
        input_circuits = self.circuit.prepare_input_circuits(inputs)

        # Convert to tfq compatible format
        return tfq.convert_to_tensor(input_circuits)

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "n_qubits": self.n_qubits,
                "measurement_indices": self.measurement_indices,
                "expectation": self.expectation,
                "n_samples": self.n_samples,
            }
        )
        return config
