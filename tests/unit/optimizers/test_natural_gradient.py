"""
Tests for the quantum natural gradient optimizer.
"""
import numpy as np
import pytest
import tensorflow as tf

from quantum_nn.optimizers import QuantumNaturalGradient


class TestQuantumNaturalGradient:
    """Test suite for quantum natural gradient optimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        # Create a simple circuit evaluator for testing
        def circuit_evaluator(params, inputs=None):
            return tf.sin(params)
        
        optimizer = QuantumNaturalGradient(
            circuit_evaluator=circuit_evaluator,
            learning_rate=0.1
        )
        
        assert optimizer.learning_rate == 0.1
        assert optimizer.damping == 0.01
    
    def test_default_fidelity_estimator(self):
        """Test the default fidelity estimator."""
        # Create a simple circuit evaluator that returns the input
        def circuit_evaluator(params, inputs=None):
            return params
        
        optimizer = QuantumNaturalGradient(
            circuit_evaluator=circuit_evaluator,
            learning_rate=0.1
        )
        
        # Test with identical parameters (fidelity should be 1)
        params1 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        params2 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        
        fidelity = optimizer._default_fidelity_estimator(params1, params2)
        assert tf.abs(fidelity - 1.0) < 1e-5
        
        # Test with orthogonal parameters (fidelity should be 0)
        params1 = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)
        params2 = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)
        
        fidelity = optimizer._default_fidelity_estimator(params1, params2)
        assert tf.abs(fidelity) < 1e-5
    
    def test_compute_fisher_matrix(self):
        """Test computation of the quantum Fisher information matrix."""
        # Create a simple circuit evaluator for testing
        def circuit_evaluator(params, inputs=None):
            return tf.nn.softmax(params)
        
        # Create a custom fidelity estimator for deterministic testing
        def fidelity_estimator(params1, params2, inputs=None):
            p1 = tf.nn.softmax(params1)
            p2 = tf.nn.softmax(params2)
            return tf.reduce_sum(tf.sqrt(p1 * p2)) ** 2
        
        optimizer = QuantumNaturalGradient(
            circuit_evaluator=circuit_evaluator,
            fidelity_estimator=fidelity_estimator,
            learning_rate=0.1
        )
        
        # Test with a 2D parameter vector
        params = tf.constant([0.0, 0.0], dtype=tf.float32)
        
        fisher_matrix = optimizer._compute_fisher_matrix(params, eps=0.1)
        
        # Check matrix properties
        assert fisher_matrix.shape == (2, 2)
        
        # Fisher matrix should be positive definite
        eigenvalues = tf.linalg.eigvalsh(fisher_matrix)
        assert tf.reduce_all(eigenvalues > 0)
        
        # Fisher matrix should be symmetric
        assert tf.reduce_all(tf.abs(fisher_matrix - tf.transpose(fisher_matrix)) < 1e-5)
    
    def test_natural_gradient(self):
        """Test natural gradient computation."""
        # Create a simple circuit evaluator for testing
        def circuit_evaluator(params, inputs=None):
            return tf.nn.softmax(params)
        
        # Create a custom fidelity estimator for deterministic testing
        def fidelity_estimator(params1, params2, inputs=None):
            p1 = tf.nn.softmax(params1)
            p2 = tf.nn.softmax(params2)
            return tf.reduce_sum(tf.sqrt(p1 * p2)) ** 2
        
        optimizer = QuantumNaturalGradient(
            circuit_evaluator=circuit_evaluator,
            fidelity_estimator=fidelity_estimator,
            learning_rate=0.1
        )
        
        # Test with a 2D parameter vector and gradient
        params = tf.constant([0.0, 0.0], dtype=tf.float32)
        grad = tf.constant([1.0, 1.0], dtype=tf.float32)
        
        natural_grad = optimizer._natural_gradient(params, grad)
        
        # Check result properties
        assert natural_grad.shape == grad.shape
        
        # Natural gradient should be different from the original gradient
        assert not tf.reduce_all(tf.abs(natural_grad - grad) < 1e-5)
