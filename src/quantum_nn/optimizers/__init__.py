"""Quantum-aware optimization techniques."""

from .natural_gradient import QuantumNaturalGradient
from .parameter_shift import ParameterShiftOptimizer
from .quantum_aware_optimizer import QuantumAwareOptimizer
from .spsa import SPSAOptimizer

__all__ = [
    "QuantumAwareOptimizer",
    "ParameterShiftOptimizer",
    "QuantumNaturalGradient",
    "SPSAOptimizer",
]
