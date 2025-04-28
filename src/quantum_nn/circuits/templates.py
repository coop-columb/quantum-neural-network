"""
Advanced quantum circuit templates for quantum neural networks.

This module provides specialized circuit architectures optimized for
different machine learning tasks and quantum computing paradigms.
"""
from typing import List, Optional, Tuple, Union, Callable

import numpy as np
import pennylane as qml


class QuantumCircuitTemplate:
    """Base class for quantum circuit templates."""
    
    def __init__(self, n_qubits: int):
        """
        Initialize a quantum circuit template.
        
        Args:
            n_qubits: Number of qubits in the circuit
        """
        self.n_qubits = n_qubits
    
    def apply(self, params: np.ndarray, wires: List[int]):
        """
        Apply the circuit template to the specified wires.
        
        Args:
            params: Circuit parameters
            wires: Quantum wires to apply the circuit to
        """
        raise NotImplementedError("Subclasses must implement apply method")
    
    def parameter_count(self) -> int:
        """
        Return the number of parameters required by this template.
        
        Returns:
            Number of parameters
        """
        raise NotImplementedError("Subclasses must implement parameter_count method")


class StronglyEntanglingLayers(QuantumCircuitTemplate):
    """
    Strongly entangling layers circuit template.
    
    This template creates a circuit with alternating rotation and entanglement layers
    that maximizes the entanglement between qubits, suitable for complex quantum
    machine learning tasks.
    """
    
    def __init__(self, n_qubits: int, n_layers: int, pattern: str = "full"):
        """
        Initialize strongly entangling layers template.
        
        Args:
            n_qubits: Number of qubits in the circuit
            n_layers: Number of entangling layers
            pattern: Entanglement pattern ('full', 'linear', 'circular', or 'nearest_neighbor')
        """
        super().__init__(n_qubits)
        self.n_layers = n_layers
        self.pattern = pattern
    
    def apply(self, params: np.ndarray, wires: List[int]):
        """
        Apply the circuit template to the specified wires.
        
        Args:
            params: Circuit parameters (shape should match parameter_count)
            wires: Quantum wires to apply the circuit to
        """
        n_params = self.parameter_count()
        if params.shape[0] != n_params:
            raise ValueError(f"Expected {n_params} parameters, got {params.shape[0]}")
        
        param_idx = 0
        
        # Apply alternating layers of rotations and entanglements
        for layer in range(self.n_layers):
            # Rotation layer - 3 rotations per qubit
            for wire_idx, wire in enumerate(wires):
                qml.Rot(params[param_idx], params[param_idx + 1], params[param_idx + 2], wires=wire)
                param_idx += 3
            
            # Entanglement layer
            if self.pattern == "full":
                # Full entanglement - each qubit entangled with every other
                for i, wire_i in enumerate(wires[:-1]):
                    for wire_j in wires[i+1:]:
                        qml.CNOT(wires=[wire_i, wire_j])
            
            elif self.pattern == "linear":
                # Linear entanglement - each qubit entangled with the next
                for i in range(len(wires) - 1):
                    qml.CNOT(wires=[wires[i], wires[i+1]])
            
            elif self.pattern == "circular":
                # Circular entanglement - like linear but with a connection back to the first qubit
                for i in range(len(wires)):
                    qml.CNOT(wires=[wires[i], wires[(i+1) % len(wires)]])
            
            elif self.pattern == "nearest_neighbor":
                # Apply entanglement between nearest neighbors with increasing range
                # At layer l, connect qubits that are l+1 apart
                range_l = (layer % (len(wires) - 1)) + 1
                for i in range(len(wires) - range_l):
                    qml.CNOT(wires=[wires[i], wires[i + range_l]])
            
            else:
                raise ValueError(f"Unknown entanglement pattern: {self.pattern}")
    
    def parameter_count(self) -> int:
        """
        Return the number of parameters required by this template.
        
        Returns:
            Number of parameters (3 rotations per qubit per layer)
        """
        return 3 * self.n_qubits * self.n_layers


class QuantumConvolutionLayers(QuantumCircuitTemplate):
    """
    Quantum convolution circuit template.
    
    This template implements a quantum analog of convolutional layers,
    applying the same unitary transformation to overlapping subsets of qubits.
    """
    
    def __init__(
        self, 
        n_qubits: int, 
        n_layers: int, 
        kernel_size: int = 2, 
        stride: int = 1
    ):
        """
        Initialize quantum convolution layers template.
        
        Args:
            n_qubits: Number of qubits in the circuit
            n_layers: Number of convolutional layers
            kernel_size: Size of the convolutional kernel (number of qubits)
            stride: Stride of the convolution operation
        """
        super().__init__(n_qubits)
        if kernel_size > n_qubits:
            raise ValueError(f"Kernel size ({kernel_size}) cannot be larger than the number of qubits ({n_qubits})")
        
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Calculate number of convolution operations per layer
        self.n_convs_per_layer = 1 + (n_qubits - kernel_size) // stride
        
    def apply(self, params: np.ndarray, wires: List[int]):
        """
        Apply the circuit template to the specified wires.
        
        Args:
            params: Circuit parameters (shape should match parameter_count)
            wires: Quantum wires to apply the circuit to
        """
        n_params = self.parameter_count()
        if params.shape[0] != n_params:
            raise ValueError(f"Expected {n_params} parameters, got {params.shape[0]}")
        
        param_idx = 0
        
        # Apply convolutional layers
        for layer in range(self.n_layers):
            # Apply convolution operations with shared parameters
            layer_params = params[param_idx:param_idx + self.params_per_kernel()]
            param_idx += self.params_per_kernel()
            
            # Apply the same unitary to each kernel position
            for conv_idx in range(self.n_convs_per_layer):
                # Calculate the wires for this convolution
                start_idx = conv_idx * self.stride
                end_idx = start_idx + self.kernel_size
                kernel_wires = wires[start_idx:end_idx]
                
                # Apply a parameterized unitary to these wires
                kernel_param_idx = 0
                
                # First apply single-qubit rotations
                for wire in kernel_wires:
                    qml.Rot(
                        layer_params[kernel_param_idx],
                        layer_params[kernel_param_idx + 1],
                        layer_params[kernel_param_idx + 2],
                        wires=wire
                    )
                    kernel_param_idx += 3
                
                # Then apply two-qubit entangling gates between adjacent qubits in the kernel
                for i in range(len(kernel_wires) - 1):
                    qml.CRot(
                        layer_params[kernel_param_idx],
                        layer_params[kernel_param_idx + 1],
                        layer_params[kernel_param_idx + 2],
                        wires=[kernel_wires[i], kernel_wires[i+1]]
                    )
                    kernel_param_idx += 3
    
    def params_per_kernel(self) -> int:
        """Calculate parameters needed per kernel."""
        # 3 parameters per single-qubit rotation for each qubit in the kernel
        single_qubit_params = 3 * self.kernel_size
        
        # 3 parameters per two-qubit gate for each adjacent pair in the kernel
        two_qubit_params = 3 * (self.kernel_size - 1)
        
        return single_qubit_params + two_qubit_params
    
    def parameter_count(self) -> int:
        """
        Return the number of parameters required by this template.
        
        Returns:
            Number of parameters
        """
        # Parameters per kernel * number of layers
        # Note: parameters are shared across kernel positions in the same layer
        return self.params_per_kernel() * self.n_layers


class QuantumResidualLayers(QuantumCircuitTemplate):
    """
    Quantum residual circuit template.
    
    This template implements a quantum analog of residual connections,
    where the quantum state at the input of a block can be combined with
    its output through controlled operations.
    """
    
    def __init__(
        self, 
        n_qubits: int, 
        n_blocks: int, 
        block_template: QuantumCircuitTemplate
    ):
        """
        Initialize quantum residual layers template.
        
        Args:
            n_qubits: Number of qubits in the circuit
            n_blocks: Number of residual blocks
            block_template: Circuit template to use for each block
        """
        super().__init__(n_qubits)
        self.n_blocks = n_blocks
        self.block_template = block_template
        
        # Verify that the block template has the same number of qubits
        if block_template.n_qubits != n_qubits:
            raise ValueError(
                f"Block template n_qubits ({block_template.n_qubits}) "
                f"must match circuit n_qubits ({n_qubits})"
            )
        
        # Parameters for the residual connections
        self.residual_params_per_block = n_qubits
    
    def apply(self, params: np.ndarray, wires: List[int]):
        """
        Apply the circuit template to the specified wires.
        
        Args:
            params: Circuit parameters (shape should match parameter_count)
            wires: Quantum wires to apply the circuit to
        """
        n_params = self.parameter_count()
        if params.shape[0] != n_params:
            raise ValueError(f"Expected {n_params} parameters, got {params.shape[0]}")
        
        param_idx = 0
        block_params_count = self.block_template.parameter_count()
        
        # Apply residual blocks
        for block in range(self.n_blocks):
            # Get parameters for this block
            block_params = params[param_idx:param_idx + block_params_count]
            param_idx += block_params_count
            
            # Apply block template
            self.block_template.apply(block_params, wires)
            
            # Apply residual connection if not the last block
            # This is done through controlled operations that can
            # partially "undo" the block transformations
            if block < self.n_blocks - 1:
                residual_params = params[
                    param_idx:param_idx + self.residual_params_per_block
                ]
                param_idx += self.residual_params_per_block
                
                # Apply parameterized residual connections
                for i, wire in enumerate(wires):
                    # Parameterized phase shift controls the residual strength
                    qml.PhaseShift(residual_params[i], wires=wire)
    
    def parameter_count(self) -> int:
        """
        Return the number of parameters required by this template.
        
        Returns:
            Number of parameters
        """
        # Parameters for blocks plus parameters for residual connections
        block_params = self.block_template.parameter_count() * self.n_blocks
        residual_params = self.residual_params_per_block * (self.n_blocks - 1)
        return block_params + residual_params
