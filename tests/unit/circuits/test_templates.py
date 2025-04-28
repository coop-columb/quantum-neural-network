"""
Tests for quantum circuit templates.
"""
import numpy as np
import pytest
import pennylane as qml

from quantum_nn.circuits.templates import (
    StronglyEntanglingLayers,
    QuantumConvolutionLayers,
    QuantumResidualLayers
)


class TestCircuitTemplates:
    """Test suite for quantum circuit templates."""

    def test_strongly_entangling_layers(self):
        """Test strongly entangling layers template."""
        # Create template
        n_qubits = 4
        n_layers = 2
        template = StronglyEntanglingLayers(n_qubits, n_layers)
        
        # Check parameter count
        expected_params = 3 * n_qubits * n_layers  # 3 rotations per qubit per layer
        assert template.parameter_count() == expected_params
        
        # Test application with random parameters
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def circuit(params):
            template.apply(params, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        params = np.random.random(template.parameter_count())
        result = circuit(params)
        
        # Check result shape
        assert len(result) == n_qubits
        
        # Check parameter validation
        with pytest.raises(ValueError):
            # Wrong number of parameters
            circuit(params[:-1])

    def test_quantum_convolution_layers(self):
        """Test quantum convolution layers template."""
        # Create template
        n_qubits = 6
        n_layers = 2
        kernel_size = 3
        template = QuantumConvolutionLayers(n_qubits, n_layers, kernel_size)
        
        # Calculate expected parameter count
        # 3 params per qubit for rotations + 3 params per adjacent pair for entanglement
        params_per_kernel = 3 * kernel_size + 3 * (kernel_size - 1)
        expected_params = params_per_kernel * n_layers
        
        assert template.parameter_count() == expected_params
        
        # Test application with random parameters
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def circuit(params):
            template.apply(params, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        params = np.random.random(template.parameter_count())
        result = circuit(params)
        
        # Check result shape
        assert len(result) == n_qubits
        
        # Check parameter validation
        with pytest.raises(ValueError):
            # Wrong number of parameters
            circuit(params[:-1])

    def test_quantum_residual_layers(self):
        """Test quantum residual layers template."""
        # Create template
        n_qubits = 4
        n_blocks = 2
        block_template = StronglyEntanglingLayers(n_qubits, n_layers=1)
        template = QuantumResidualLayers(n_qubits, n_blocks, block_template)
        
        # Calculate expected parameter count
        block_params = block_template.parameter_count() * n_blocks
        residual_params = n_qubits * (n_blocks - 1)
        expected_params = block_params + residual_params
        
        assert template.parameter_count() == expected_params
        
        # Test application with random parameters
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def circuit(params):
            template.apply(params, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        params = np.random.random(template.parameter_count())
        result = circuit(params)
        
        # Check result shape
        assert len(result) == n_qubits
        
        # Check parameter validation
        with pytest.raises(ValueError):
            # Wrong number of parameters
            circuit(params[:-1])
