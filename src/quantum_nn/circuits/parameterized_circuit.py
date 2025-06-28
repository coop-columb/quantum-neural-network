"""
Parameterized quantum circuit implementation.

This module provides the fundamental quantum circuit structures
that serve as the computational backbone for quantum neural networks.
"""

from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import pennylane as qml

from quantum_nn.circuits.templates import (
    QuantumCircuitTemplate,
    StronglyEntanglingLayers,
    QuantumConvolutionLayers,
    QuantumResidualLayers,
)
from quantum_nn.circuits.encodings import (
    QuantumEncoder,
    AmplitudeEncoding,
    AngleEncoding,
    BasisEncoding,
    HybridEncoding,
)


class ParameterizedCircuit:
    """
    A parameterized quantum circuit for quantum neural networks.

    This class provides a flexible interface for creating and managing
    parameterized quantum circuits with customizable structure and complexity.

    Attributes:
        n_qubits: Number of qubits in the circuit
        encoder: Quantum encoder for data input
        template: Quantum circuit template
        observables: Quantum observables to measure
        device: Quantum device to use for simulation
    """

    def __init__(
        self,
        n_qubits: int = 4,
        template: Optional[Union[str, QuantumCircuitTemplate]] = None,
        template_kwargs: Optional[Dict[str, Any]] = None,
        encoder: Optional[Union[str, QuantumEncoder]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        observables: Optional[List[str]] = None,
        device: str = "default.qubit",
    ):
        """
        Initialize a parameterized quantum circuit.

        Args:
            n_qubits: Number of qubits in the circuit
            template: Circuit template or name ('strongly_entangling', 'convolution', 'residual')
            template_kwargs: Additional arguments for the template
            encoder: Data encoder or name ('amplitude', 'angle', 'basis', 'hybrid')
            encoder_kwargs: Additional arguments for the encoder
            observables: Quantum observables to measure (default: Pauli-Z on all qubits)
            device: Quantum device to use for simulation
        """
        self.n_qubits = n_qubits

        # Set up encoder
        self.encoder = self._create_encoder(encoder, encoder_kwargs)

        # Set up template
        self.template = self._create_template(template, template_kwargs)

        # Set up observables
        if observables is None:
            self.observables = [qml.PauliZ(i) for i in range(n_qubits)]
        else:
            self.observables = [
                getattr(qml, obs)(i) for i, obs in enumerate(observables)
            ]

        # Set up device and circuit
        self.device = qml.device(device, wires=n_qubits)
        self.circuit = qml.QNode(self._circuit_fn, self.device)

        # Calculate number of parameters
        if self.template:
            self.n_params = self.template.parameter_count()
        else:
            # Default to 3 params per qubit if no template
            self.n_params = 3 * n_qubits

    def _create_encoder(
        self,
        encoder: Optional[Union[str, QuantumEncoder]],
        kwargs: Optional[Dict[str, Any]],
    ) -> Optional[QuantumEncoder]:
        """Create a quantum encoder from name or instance."""
        if encoder is None:
            return AngleEncoding(self.n_qubits, rotation="X")

        if isinstance(encoder, QuantumEncoder):
            return encoder

        kwargs = kwargs or {}

        if encoder.lower() == "amplitude":
            return AmplitudeEncoding(self.n_qubits, **kwargs)
        elif encoder.lower() == "angle":
            return AngleEncoding(self.n_qubits, **kwargs)
        elif encoder.lower() == "basis":
            return BasisEncoding(self.n_qubits, **kwargs)
        elif encoder.lower() == "hybrid":
            # For hybrid, we need to create sub-encoders
            sub_encoders = kwargs.pop(
                "encoders",
                [
                    AngleEncoding(self.n_qubits // 2),
                    AmplitudeEncoding(self.n_qubits - (self.n_qubits // 2)),
                ],
            )
            return HybridEncoding(self.n_qubits, sub_encoders, **kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder}")

    def _create_template(
        self,
        template: Optional[Union[str, QuantumCircuitTemplate]],
        kwargs: Optional[Dict[str, Any]],
    ) -> Optional[QuantumCircuitTemplate]:
        """Create a quantum circuit template from name or instance."""
        if template is None:
            return StronglyEntanglingLayers(self.n_qubits, n_layers=2)

        if isinstance(template, QuantumCircuitTemplate):
            return template

        kwargs = kwargs or {}

        if template.lower() == "strongly_entangling":
            return StronglyEntanglingLayers(self.n_qubits, **kwargs)
        elif template.lower() == "convolution":
            return QuantumConvolutionLayers(self.n_qubits, **kwargs)
        elif template.lower() == "residual":
            # For residual, we need to create a sub-template
            block_template = kwargs.pop(
                "block_template", StronglyEntanglingLayers(self.n_qubits, n_layers=1)
            )
            return QuantumResidualLayers(
                self.n_qubits, block_template=block_template, **kwargs
            )
        else:
            raise ValueError(f"Unknown template type: {template}")

    def _circuit_fn(self, params, inputs=None):
        """
        Define the quantum circuit function.

        Args:
            params: Circuit parameters for rotation gates
            inputs: Classical input data to encode (optional)

        Returns:
            Measurement results from the circuit
        """
        # Data encoding layer
        if inputs is not None and self.encoder:
            self.encoder.encode(inputs, wires=range(self.n_qubits))

        # Apply circuit template
        if self.template:
            self.template.apply(params, wires=range(self.n_qubits))
        else:
            # Fallback to simple rotations if no template
            param_idx = 0
            for i in range(self.n_qubits):
                qml.RX(params[param_idx], wires=i)
                param_idx += 1
                qml.RY(params[param_idx], wires=i)
                param_idx += 1
                qml.RZ(params[param_idx], wires=i)
                param_idx += 1

        # Return measurement results
        return [qml.expval(obs) for obs in self.observables]

    def __call__(
        self, params: np.ndarray, inputs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Execute the quantum circuit with the given parameters and inputs.

        Args:
            params: Circuit parameters
            inputs: Classical input data (optional)

        Returns:
            Circuit measurement results
        """
        return np.array(self.circuit(params, inputs))

    def get_n_params(self) -> int:
        """
        Get the total number of parameters in the circuit.

        Returns:
            Number of parameters
        """
        return self.n_params
