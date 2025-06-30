"""
Tests for the SPSA optimizer.
"""

import numpy as np
import pytest
import tensorflow as tf

from quantum_nn.optimizers import SPSAOptimizer


class TestSPSAOptimizer:
    """Test suite for SPSA optimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""

        # Create a simple circuit evaluator for testing
        def circuit_evaluator(params, inputs=None):
            return (params - 1.0) ** 2

        optimizer = SPSAOptimizer(
            circuit_evaluator=circuit_evaluator, learning_rate=0.1
        )

        assert optimizer.learning_rate == 0.1
        assert optimizer.perturbation == 0.01
        assert optimizer.a == 0.01
        assert optimizer.c == 0.1
        assert optimizer.alpha == 0.602
        assert optimizer.gamma == 0.101
        assert optimizer.momentum == 0.9

    def test_spsa_gradient(self):
        """Test gradient approximation using SPSA."""

        # Create a quadratic function with minimum at [1.0, 1.0]
        def circuit_evaluator(params, inputs=None):
            return tf.reduce_sum((params - 1.0) ** 2)

        optimizer = SPSAOptimizer(
            circuit_evaluator=circuit_evaluator, learning_rate=0.1, a=0.1, c=0.1
        )

        # Initialize iteration counter for deterministic behavior in test
        optimizer._iteration.assign(0)

        # Test at [0.0, 0.0], gradient should point towards [1.0, 1.0]
        params = tf.constant([0.0, 0.0], dtype=tf.float32)

        # Run multiple trials since SPSA is stochastic
        # We expect the average gradient to point roughly towards [1.0, 1.0]
        n_trials = 10
        gradients = []

        for _ in range(n_trials):
            grad = optimizer._spsa_gradient(params)
            gradients.append(grad.numpy())

        # Average gradient
        avg_gradient = np.mean(gradients, axis=0)

        # Sign should be negative (pointing towards minimum)
        assert avg_gradient[0] < 0
        assert avg_gradient[1] < 0

    def test_optimization(self):
        """Test optimization of a simple quadratic function."""

        # Create a quadratic function with minimum at [1.0, 1.0]
        def circuit_evaluator(params, inputs=None):
            return tf.reduce_sum((params - 1.0) ** 2)

        # Due to the stochastic nature of SPSA, use larger learning rate
        # and fixed random seed for reproducibility
        tf.random.set_seed(42)

        optimizer = SPSAOptimizer(
            circuit_evaluator=circuit_evaluator,
            learning_rate=0.2,
            a=0.2,
            c=0.1,
            momentum=0.5,
        )

        # Initialize variable at [0.0, 0.0]
        var = tf.Variable([0.0, 0.0], dtype=tf.float32)

        # Create velocity slot
        optimizer._velocity[var.ref()] = tf.Variable(
            tf.zeros_like(var), trainable=False
        )

        # Run optimization for several iterations
        n_iterations = 20
        for _ in range(n_iterations):
            # Apply SPSA update directly
            grad = optimizer._spsa_gradient(var)

            # Update velocity and variable
            velocity = optimizer._velocity[var.ref()]
            velocity.assign(
                optimizer.momentum * velocity + (1 - optimizer.momentum) * grad
            )
            var.assign_sub(optimizer.learning_rate * velocity)

        # Check that the variable moved closer to the minimum
        # Since SPSA is stochastic, we can't expect exact convergence
        # but should be much closer than initial position
        distance = tf.sqrt(tf.reduce_sum((var - 1.0) ** 2))
        assert (
            distance < 0.5
        )  # Much closer to [1.0, 1.0] than initial distance of 1.414
