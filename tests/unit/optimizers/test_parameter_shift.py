"""
Tests for the parameter-shift optimizer.
"""
import numpy as np
import pytest
import tensorflow as tf

from quantum_nn.optimizers import ParameterShiftOptimizer


class TestParameterShiftOptimizer:
    """Test suite for parameter-shift optimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        # Create a simple circuit evaluator for testing
        def circuit_evaluator(params, inputs=None):
            return tf.sin(params)
        
        optimizer = ParameterShiftOptimizer(
            circuit_evaluator=circuit_evaluator,
            learning_rate=0.1
        )
        
        assert optimizer.learning_rate == 0.1
        assert optimizer.shift == np.pi/2
        assert optimizer.beta_1 == 0.9
        assert optimizer.beta_2 == 0.999
        assert optimizer.epsilon == 1e-7
    
    def test_parameter_shift_gradient(self):
        """Test gradient computation using parameter-shift rule."""
        # Create a simple circuit evaluator that computes sin(x)
        # The gradient should be cos(x)
        def circuit_evaluator(params, inputs=None):
            return tf.sin(params)
        
        optimizer = ParameterShiftOptimizer(
            circuit_evaluator=circuit_evaluator,
            learning_rate=0.1
        )
        
        # Test at x = 0, where cos(0) = 1
        params = tf.constant([0.0], dtype=tf.float32)
        gradient = optimizer._parameter_shift_gradient(params)
        
        assert tf.abs(gradient[0] - 1.0) < 1e-5
        
        # Test at x = pi/2, where cos(pi/2) = 0
        params = tf.constant([np.pi/2], dtype=tf.float32)
        gradient = optimizer._parameter_shift_gradient(params)
        
        assert tf.abs(gradient[0]) < 1e-5
        
        # Test at x = pi, where cos(pi) = -1
        params = tf.constant([np.pi], dtype=tf.float32)
        gradient = optimizer._parameter_shift_gradient(params)
        
        assert tf.abs(gradient[0] + 1.0) < 1e-5
    
    def test_optimization(self):
        """Test optimization of a simple function."""
        # Create a simple circuit evaluator for a function with minimum at x = 1
        def circuit_evaluator(params, inputs=None):
            return (params - 1.0) ** 2
        
        optimizer = ParameterShiftOptimizer(
            circuit_evaluator=circuit_evaluator,
            learning_rate=0.1
        )
        
        # Initialize variable at x = 0
        var = tf.Variable([0.0], dtype=tf.float32)
        
        # Run a few optimization steps
        for _ in range(10):
            # Manually compute gradient for this test
            with tf.GradientTape() as tape:
                loss = circuit_evaluator(var)
            
            grad = tape.gradient(loss, var)
            optimizer.apply_gradients([(grad, var)])
        
        # Check that the variable moved closer to the minimum
        assert tf.abs(var[0] - 1.0) < 0.1
