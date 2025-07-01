"""
Base classes for quantum neural network circuits.

This module provides the foundational abstract classes for creating parameterized
quantum circuit templates used in quantum neural networks. These templates define
reusable circuit patterns that can be applied to quantum devices.
"""

from typing import List

import numpy as np


class QuantumCircuitTemplate:
    """
    Abstract base class for quantum circuit templates.

    This class defines the interface for parameterized quantum circuit templates
    that can be used as building blocks in quantum neural networks. Templates
    encapsulate specific circuit patterns (e.g., strongly entangling layers,
    convolution-like operations) and can be applied to different sets of qubits
    with varying parameters.

    The template pattern allows for:
    - Reusable circuit components
    - Parameterized quantum operations
    - Modular circuit design
    - Easy experimentation with different architectures

    Attributes:
        n_qubits (int): Number of qubits this template operates on

    Example:
        >>> # Custom template implementation
        >>> class MyTemplate(QuantumCircuitTemplate):
        ...     def apply(self, params, wires):
        ...         # Apply circuit operations
        ...         pass
        ...     def parameter_count(self):
        ...         return 3 * self.n_qubits
        >>>
        >>> template = MyTemplate(n_qubits=4)
        >>> params = np.random.random(template.parameter_count())
        >>> template.apply(params, wires=[0, 1, 2, 3])
    """

    def __init__(self, n_qubits: int):
        """
        Initialize the quantum circuit template.

        Args:
            n_qubits (int): Number of qubits this template will operate on.
                           Must be a positive integer.

        Raises:
            ValueError: If n_qubits is not a positive integer.
        """
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError(f"n_qubits must be a positive integer, got {n_qubits}")

        self.n_qubits = n_qubits

    def apply(self, params: np.ndarray, wires: List[int]) -> None:
        """
        Apply the quantum circuit template to the specified wires.

        This method must be implemented by subclasses to define the specific
        quantum operations that make up the template. The implementation should
        use the provided parameters to create a parameterized quantum circuit
        on the given wires.

        Args:
            params (np.ndarray): Array of parameters for the quantum gates.
                               Length must match the value returned by parameter_count().
            wires (List[int]): List of wire indices to apply the circuit to.
                              Length should typically match n_qubits.

        Raises:
            NotImplementedError: This is an abstract method that must be
                               implemented by subclasses.
            ValueError: If the number of parameters or wires is incorrect.

        Note:
            Implementations should validate that len(params) == parameter_count()
            and that the wires list has the expected length for the template.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the apply method. "
            "This method should apply the quantum circuit template to the "
            "specified wires using the provided parameters."
        )

    def parameter_count(self) -> int:
        """
        Return the total number of parameters required by this template.

        This method must be implemented by subclasses to specify how many
        parameters are needed to fully parameterize the quantum circuit
        template. This count is used for parameter initialization and
        validation.

        Returns:
            int: The number of parameters required by this template.
                Must be a non-negative integer.

        Raises:
            NotImplementedError: This is an abstract method that must be
                               implemented by subclasses.

        Note:
            The returned value should be deterministic and based only on
            the template's configuration (e.g., n_qubits, n_layers).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the parameter_count method. "
            "This method should return the total number of parameters required "
            "by the quantum circuit template."
        )
