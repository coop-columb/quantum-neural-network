"""
Quantum-aware optimization techniques for quantum neural networks.

This module provides specialized optimization algorithms designed for training
quantum neural networks and variational quantum algorithms. Unlike classical
optimizers, these quantum-aware optimizers account for the unique challenges
of quantum computing, including:

- Limited gradient information due to measurement constraints
- Noise in quantum hardware implementations
- The geometric structure of quantum state spaces
- Compatibility with quantum gradient computation methods

Available Optimizers:
===================

QuantumAwareOptimizer
    Abstract base class for all quantum-aware optimizers. Extends TensorFlow's
    optimizer interface with quantum-specific functionality.

ParameterShiftOptimizer
    Implements the parameter-shift rule for exact gradient computation.
    Compatible with quantum hardware and provides exact gradients for
    quantum circuits with appropriate gate sets.

QuantumNaturalGradient
    Uses the quantum Fisher information matrix to perform optimization
    in the natural geometry of quantum state space. Often provides
    faster convergence than standard gradient descent.

SPSAOptimizer
    Simultaneous Perturbation Stochastic Approximation optimizer.
    Highly efficient for noisy quantum systems, requiring only 2 function
    evaluations per iteration regardless of parameter count.

Usage Examples:
==============

Basic parameter-shift optimization:

    >>> from quantum_nn.optimizers import ParameterShiftOptimizer
    >>>
    >>> def circuit_loss(params, inputs):
    ...     # Evaluate quantum circuit and return loss
    ...     return expectation_value
    >>>
    >>> optimizer = ParameterShiftOptimizer(
    ...     circuit_evaluator=circuit_loss,
    ...     learning_rate=0.01
    ... )

Natural gradient optimization:

    >>> from quantum_nn.optimizers import QuantumNaturalGradient
    >>>
    >>> optimizer = QuantumNaturalGradient(
    ...     circuit_evaluator=circuit_loss,
    ...     learning_rate=0.01,
    ...     damping=0.001
    ... )

SPSA for noisy quantum hardware:

    >>> from quantum_nn.optimizers import SPSAOptimizer
    >>>
    >>> optimizer = SPSAOptimizer(
    ...     circuit_evaluator=noisy_circuit_loss,
    ...     learning_rate=0.1,
    ...     c=0.1  # Perturbation size proportional to noise
    ... )

Notes:
======
- All optimizers are compatible with TensorFlow's training loops
- For quantum hardware deployment, SPSA is often most robust to noise
- Parameter-shift provides exact gradients but requires more evaluations
- Natural gradient often converges faster but has higher computational cost
- Choose optimizer based on circuit size, noise level, and convergence requirements

References:
===========
- Cerezo et al. "Variational quantum algorithms" Nature Reviews Physics 3, 625–644 (2021)
- Schuld & Petruccione "Supervised learning with quantum computers" Springer (2018)
- McClean et al. "The theory of variational hybrid quantum-classical algorithms"
  New Journal of Physics 18, 023023 (2016)
"""

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
