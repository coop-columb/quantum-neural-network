"""
Production-ready quantum layer using PennyLane with actual quantum circuits.
"""

from typing import Callable, List, Optional

import numpy as np
import pennylane as qml
import tensorflow as tf


class PennyLaneQuantumLayer(tf.keras.layers.Layer):
    """
    Production quantum layer with real quantum circuit execution.

    This implementation:
    - Uses actual PennyLane quantum circuits
    - Implements parameter-shift gradients
    - Supports multiple quantum backends
    - Includes noise simulation options
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        backend: str = "default.qubit",
        diff_method: str = "parameter-shift",
        encoding_method: str = "angle",
        add_noise: bool = False,
        noise_prob: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend
        self.diff_method = diff_method
        self.encoding_method = encoding_method
        self.add_noise = add_noise
        self.noise_prob = noise_prob

        # Create quantum device
        self.dev = qml.device(backend, wires=n_qubits)

        # Create the quantum circuit
        self.qnode = qml.QNode(self._quantum_circuit, self.dev, diff_method=diff_method)

    def build(self, input_shape):
        """Build the layer with quantum parameters."""
        # Parameters for rotation gates
        self.theta = self.add_weight(
            name="theta",
            shape=(self.n_layers, self.n_qubits, 3),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def _quantum_circuit(self, inputs, weights):
        """Define the parameterized quantum circuit."""
        # Data encoding
        if self.encoding_method == "angle":
            for i in range(min(len(inputs), self.n_qubits)):
                qml.RY(inputs[i], wires=i)
        elif self.encoding_method == "amplitude":
            qml.AmplitudeEmbedding(
                inputs[: 2**self.n_qubits], wires=range(self.n_qubits)
            )

        # Variational layers
        for l in range(self.n_layers):
            # Rotation layer
            for i in range(self.n_qubits):
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if self.n_qubits > 2:
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            # Optional noise
            if self.add_noise:
                for i in range(self.n_qubits):
                    qml.DepolarizingChannel(self.noise_prob, wires=i)

        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def call(self, inputs):
        """Execute the quantum circuit for batch of inputs."""
        batch_size = tf.shape(inputs)[0]
        outputs = []

        # Process each sample (quantum circuits process one at a time)
        for i in range(batch_size):
            out = self.qnode(inputs[i], self.theta)
            outputs.append(out)

        return tf.stack(outputs)


# Quick test
if __name__ == "__main__":
    print("🧪 Testing PennyLane Quantum Layer...")

    layer = PennyLaneQuantumLayer(n_qubits=4, n_layers=2)
    x = tf.random.normal((3, 8))

    output = layer(x)
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Output: {output.numpy()}")
