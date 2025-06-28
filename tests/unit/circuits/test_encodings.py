"""
Tests for quantum data encodings.
"""

import numpy as np
import pennylane as qml
import pytest

from quantum_nn.circuits.encodings import (
    AmplitudeEncoding,
    AngleEncoding,
    BasisEncoding,
    HybridEncoding,
)


class TestQuantumEncodings:
    """Test suite for quantum data encodings."""

    def test_amplitude_encoding(self):
        """Test amplitude encoding."""
        # Create encoder
        n_qubits = 3
        encoder = AmplitudeEncoding(n_qubits)

        # Test encoding with random data
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(data):
            encoder.encode(data, wires=range(n_qubits))
            return qml.state()

        # Create normalized random data
        data = np.random.random(2**n_qubits)
        data = data / np.linalg.norm(data)

        state = circuit(data)

        # Check that the state amplitudes match the input data
        assert np.allclose(np.abs(state) ** 2, data**2, atol=1e-6)

        # Test with data dimensionality validation
        with pytest.raises(ValueError):
            # Too much data
            circuit(np.random.random(2**n_qubits + 1))

    def test_angle_encoding(self):
        """Test angle encoding."""
        # Create encoder
        n_qubits = 4
        encoder = AngleEncoding(n_qubits, rotation="X")

        # Test encoding with various rotations
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit_x(data):
            encoder.encode(data, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Test with pi rotations which should flip the qubits
        data = np.array([np.pi] * n_qubits)
        result = circuit_x(data)

        # RX(pi) should flip the state from |0⟩ to |1⟩, giving -1 expectation for PauliZ
        assert np.allclose(result, [-1] * n_qubits)

        # Test with different rotation
        encoder_y = AngleEncoding(n_qubits, rotation="Y")

        @qml.qnode(dev)
        def circuit_y(data):
            encoder_y.encode(data, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # RY(pi) should also flip the state
        result = circuit_y(data)
        assert np.allclose(result, [-1] * n_qubits)

        # Test with scaling
        encoder_scaled = AngleEncoding(n_qubits, rotation="X", scaling=0.5)

        @qml.qnode(dev)
        def circuit_scaled(data):
            encoder_scaled.encode(data, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # RX(pi/2) should give 0 expectation for PauliZ
        result = circuit_scaled(data)
        assert np.allclose(result, [0] * n_qubits, atol=1e-6)

    def test_basis_encoding(self):
        """Test basis encoding."""
        # Create encoder
        n_qubits = 4
        encoder = BasisEncoding(n_qubits)

        # Test encoding with binary data
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(data):
            encoder.encode(data, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Test with alternating 0 and 1
        data = np.array([0, 1, 0, 1])
        result = circuit(data)

        # PauliZ expectation should be 1 for |0⟩ and -1 for |1⟩
        expected = [1, -1, 1, -1]
        assert np.allclose(result, expected)

        # Test with all ones
        data = np.array([1, 1, 1, 1])
        result = circuit(data)
        assert np.allclose(result, [-1] * n_qubits)

    def test_hybrid_encoding(self):
        """Test hybrid encoding."""
        # Create component encoders
        n_qubits = 4
        n_qubits_1 = 2
        n_qubits_2 = 2

        encoder1 = AngleEncoding(n_qubits_1)
        encoder2 = BasisEncoding(n_qubits_2)

        # Create hybrid encoder
        hybrid_encoder = HybridEncoding(
            n_qubits,
            [encoder1, encoder2],
            features_per_encoder=[n_qubits_1, n_qubits_2],
        )

        # Test encoding with mixed data
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(data):
            hybrid_encoder.encode(data, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # First part: angle encoding with pi (flip)
        # Second part: basis encoding with [1, 1] (flip)
        data = np.array([np.pi, np.pi, 1, 1])
        result = circuit(data)

        # All qubits should be flipped
        assert np.allclose(result, [-1] * n_qubits)

        # Test with different data combinations
        data = np.array([0, 0, 0, 1])
        result = circuit(data)

        # First two qubits: no rotation (|0⟩)
        # Last two qubits: first no flip, second flip
        expected = [1, 1, 1, -1]
        assert np.allclose(result, expected)
