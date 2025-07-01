"""
Base classes for quantum neural network circuits.
"""

from typing import List

import numpy as np


class QuantumCircuitTemplate:
    """Base class for quantum circuit templates."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def apply(self, params: np.ndarray, wires: List[int]):
        raise NotImplementedError("Subclasses must implement apply method")

    def parameter_count(self) -> int:
        raise NotImplementedError("Subclasses must implement parameter_count method")
