"""
Quantum data encoding schemes.

This module provides different methods to encode classical data into quantum states,
which is a crucial aspect of quantum machine learning.
"""
from typing import List, Optional, Callable, Union

import numpy as np
import pennylane as qml


class QuantumEncoder:
    """Base class for quantum data encoders."""
    
    def __init__(self, n_qubits: int):
        """
        Initialize a quantum encoder.
        
        Args:
            n_qubits: Number of qubits in the circuit
        """
        self.n_qubits = n_qubits
    
    def encode(self, data: np.ndarray, wires: List[int]):
        """
        Encode classical data into quantum states.
        
        Args:
            data: Classical data to encode
            wires: Quantum wires to encode the data into
        """
        raise NotImplementedError("Subclasses must implement encode method")


class AmplitudeEncoding(QuantumEncoder):
    """
    Amplitude encoding scheme.
    
    This encoder maps classical data directly into the amplitudes of
    the quantum state, providing an exponentially compact representation.
    """
    
    def __init__(self, n_qubits: int, normalize: bool = True):
        """
        Initialize an amplitude encoder.
        
        Args:
            n_qubits: Number of qubits in the circuit
            normalize: Whether to normalize the input data
        """
        super().__init__(n_qubits)
        self.normalize = normalize
        self.dimension = 2**n_qubits
    
    def encode(self, data: np.ndarray, wires: List[int]):
        """
        Encode classical data into quantum amplitudes.
        
        Args:
            data: Classical data to encode (must have 2^n_qubits elements)
            wires: Quantum wires to encode the data into
        """
        if len(data) > self.dimension:
            raise ValueError(
                f"Data dimension ({len(data)}) exceeds encoder capacity (2^{self.n_qubits}={self.dimension})"
            )
        
        # Pad data if necessary
        if len(data) < self.dimension:
            padded_data = np.zeros(self.dimension)
            padded_data[:len(data)] = data
            data = padded_data
        
        # Normalize if requested
        if self.normalize:
            norm = np.linalg.norm(data)
            if norm > 0:
                data = data / norm
        
        # Use PennyLane's built-in state preparation
        qml.AmplitudeEmbedding(data, wires=wires, normalize=False, pad_with=0.0)


class AngleEncoding(QuantumEncoder):
    """
    Angle encoding scheme.
    
    This encoder maps classical data into rotation angles of
    quantum gates, which is more directly trainable.
    """
    
    def __init__(
        self, 
        n_qubits: int, 
        rotation: str = "X", 
        scaling: Optional[float] = None
    ):
        """
        Initialize an angle encoder.
        
        Args:
            n_qubits: Number of qubits in the circuit
            rotation: Rotation gate to use ('X', 'Y', 'Z', or 'all')
            scaling: Optional scaling factor for the angles
        """
        super().__init__(n_qubits)
        self.rotation = rotation
        self.scaling = scaling
    
    def encode(self, data: np.ndarray, wires: List[int]):
        """
        Encode classical data into rotation angles.
        
        Args:
            data: Classical data to encode (one value per qubit)
            wires: Quantum wires to encode the data into
        """
        if len(data) > len(wires):
            raise ValueError(
                f"Data dimension ({len(data)}) exceeds number of qubits ({len(wires)})"
            )
        
        # Apply scaling if specified
        if self.scaling is not None:
            data = data * self.scaling
        
        # Encode data using rotation gates
        for i, (wire, value) in enumerate(zip(wires, data)):
            if self.rotation.upper() == "X" or self.rotation.upper() == "ALL":
                qml.RX(value, wires=wire)
            if self.rotation.upper() == "Y" or self.rotation.upper() == "ALL":
                qml.RY(value, wires=wire)
            if self.rotation.upper() == "Z" or self.rotation.upper() == "ALL":
                qml.RZ(value, wires=wire)


class BasisEncoding(QuantumEncoder):
    """
    Basis encoding scheme.
    
    This encoder maps binary or discrete data into computational basis states,
    which is useful for combinatorial optimization problems.
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize a basis encoder.
        
        Args:
            n_qubits: Number of qubits in the circuit
        """
        super().__init__(n_qubits)
    
    def encode(self, data: np.ndarray, wires: List[int]):
        """
        Encode binary data into basis states.
        
        Args:
            data: Binary data to encode (0 or 1 for each qubit)
            wires: Quantum wires to encode the data into
        """
        if len(data) > len(wires):
            raise ValueError(
                f"Data dimension ({len(data)}) exceeds number of qubits ({len(wires)})"
            )
        
        # Apply X gates to flip qubits from |0⟩ to |1⟩ where data is 1
        for i, (wire, value) in enumerate(zip(wires, data)):
            if value == 1 or value is True:
                qml.PauliX(wires=wire)


class HybridEncoding(QuantumEncoder):
    """
    Hybrid encoding scheme.
    
    This encoder combines multiple encoding strategies for richer
    quantum feature representation.
    """
    
    def __init__(
        self, 
        n_qubits: int, 
        encoders: List[QuantumEncoder], 
        features_per_encoder: Optional[List[int]] = None
    ):
        """
        Initialize a hybrid encoder.
        
        Args:
            n_qubits: Number of qubits in the circuit
            encoders: List of encoders to combine
            features_per_encoder: Number of features to use for each encoder
        """
        super().__init__(n_qubits)
        self.encoders = encoders
        
        if features_per_encoder is None:
            # Divide features equally among encoders
            self.features_per_encoder = [n_qubits // len(encoders)] * len(encoders)
            self.features_per_encoder[-1] += n_qubits % len(encoders)
        else:
            if sum(features_per_encoder) != n_qubits:
                raise ValueError(
                    f"Sum of features_per_encoder ({sum(features_per_encoder)}) "
                    f"must equal n_qubits ({n_qubits})"
                )
            self.features_per_encoder = features_per_encoder
    
    def encode(self, data: np.ndarray, wires: List[int]):
        """
        Encode data using multiple encoding strategies.
        
        Args:
            data: Data to encode (must match total number of features)
            wires: Quantum wires to encode the data into
        """
        if len(data) < sum(self.features_per_encoder):
            raise ValueError(
                f"Data dimension ({len(data)}) is less than total features "
                f"({sum(self.features_per_encoder)})"
            )
        
        # Split data and wires for each encoder
        start_idx = 0
        wire_idx = 0
        
        for encoder, n_features in zip(self.encoders, self.features_per_encoder):
            # Extract data for this encoder
            encoder_data = data[start_idx:start_idx + n_features]
            start_idx += n_features
            
            # Extract wires for this encoder
            encoder_wires = wires[wire_idx:wire_idx + encoder.n_qubits]
            wire_idx += encoder.n_qubits
            
            # Apply the encoder
            encoder.encode(encoder_data, encoder_wires)
