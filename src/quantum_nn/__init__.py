"""
Quantum Neural Network Framework

A comprehensive framework for implementing quantum neural networks
with TensorFlow and PennyLane integration.
"""

__version__ = "0.1.0"

# Core imports
from .layers import QuantumLayer
from .models import QuantumModel

__all__ = [
    "QuantumLayer",
    "QuantumModel",
]
