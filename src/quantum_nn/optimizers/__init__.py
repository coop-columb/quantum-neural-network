"""Quantum-aware optimization techniques."""

from .quantum_aware_optimizer import QuantumAwareOptimizer
from .parameter_shift import ParameterShiftOptimizer
from .natural_gradient import QuantumNaturalGradient
from .spsa import SPSAOptimizer

__all__ = [
    "QuantumAwareOptimizer",
    "ParameterShiftOptimizer",
    "QuantumNaturalGradient",
    "SPSAOptimizer"
]
