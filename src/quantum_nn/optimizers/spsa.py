"""
Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

This module implements the SPSA algorithm, which is particularly
well-suited for optimizing quantum circuits on noisy hardware.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from quantum_nn.optimizers.quantum_aware_optimizer import QuantumAwareOptimizer


class SPSAOptimizer(QuantumAwareOptimizer):
    """
    Optimizer using Simultaneous Perturbation Stochastic Approximation.

    SPSA is particularly effective for quantum circuit optimization
    on noisy hardware, as it requires fewer function evaluations
    than traditional gradient-based methods.
    """

    def __init__(
        self,
        circuit_evaluator: Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor],
        perturbation: float = 0.01,
        a: float = 0.01,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        learning_rate: Union[
            float, tf.keras.optimizers.schedules.LearningRateSchedule
        ] = 0.01,
        momentum: float = 0.9,
        name: str = "SPSAOptimizer",
        **kwargs,
    ):
        """
        Initialize an SPSA optimizer.

        Args:
            circuit_evaluator: Function to evaluate the quantum circuit
            perturbation: Initial perturbation magnitude
            a: Scale parameter for step size
            c: Scale parameter for perturbation size
            alpha: Decay parameter for step size
            gamma: Decay parameter for perturbation size
            learning_rate: Initial learning rate or schedule
            momentum: Momentum coefficient
            name: Name of the optimizer
            **kwargs: Additional arguments to pass to the optimizer
        """
        super().__init__(name=name, learning_rate=learning_rate, **kwargs)
        self.circuit_evaluator = circuit_evaluator
        self.perturbation = perturbation
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.momentum = momentum

        # Initialize optimizer state
        self._iteration = tf.Variable(0, name="iteration", dtype=tf.int64)
        self._velocity = {}  # Momentum velocity

    def _create_slots(self, var_list: List[tf.Variable]):
        """Create optimizer state variables."""
        for var in var_list:
            self._velocity[var.ref()] = tf.Variable(
                tf.zeros_like(var), name=f"{var.name}/velocity", trainable=False
            )

    @tf.function
    def _spsa_gradient(
        self, params: tf.Tensor, inputs: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Compute gradient approximation using SPSA.

        Args:
            params: Quantum circuit parameters
            inputs: Input data (optional)

        Returns:
            Gradient approximation
        """
        # Calculate current iteration-dependent coefficients
        iteration = tf.cast(self._iteration, tf.float32)
        a_k = self.a / tf.pow(iteration + 1, self.alpha)
        c_k = self.c / tf.pow(iteration + 1, self.gamma)

        # Generate random perturbation direction
        delta = tf.sign(tf.random.uniform(params.shape, minval=-1, maxval=1))

        # Perturb parameters in both directions
        params_plus = params + c_k * delta
        params_minus = params - c_k * delta

        # Evaluate circuit with perturbed parameters
        loss_plus = tf.reduce_mean(self.circuit_evaluator(params_plus, inputs))
        loss_minus = tf.reduce_mean(self.circuit_evaluator(params_minus, inputs))

        # Compute gradient approximation
        gradient = (loss_plus - loss_minus) / (2 * c_k * delta)

        return gradient

    def apply_gradients(
        self,
        grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]],
        name: Optional[str] = None,
        **kwargs,
    ) -> tf.Operation:
        """
        Apply SPSA gradient approximation to variables.

        Args:
            grads_and_vars: List of (gradient, variable) pairs
            name: Name of the operation
            **kwargs: Additional arguments

        Returns:
            An operation that applies the gradients
        """
        # Create slots if they don't exist
        var_list = [var for _, var in grads_and_vars]
        if not self._velocity:
            self._create_slots(var_list)

        # Increment iteration counter
        self._iteration.assign_add(1)

        # Get current learning rate
        lr = self._get_hyper("learning_rate", tf.float32)

        # Apply updates to each variable
        for grad, var in grads_and_vars:
            if grad is None:
                continue

            # For quantum layers, compute gradients using SPSA
            if var.name.startswith("quantum"):
                inputs = kwargs.get("inputs")
                grad = self._spsa_gradient(var, inputs)

            # Apply momentum
            velocity = self._velocity[var.ref()]
            velocity.assign(self.momentum * velocity + (1 - self.momentum) * grad)

            # Apply update
            var.assign_sub(lr * velocity)

        return tf.no_op(name=name)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the optimizer."""
        config = super().get_config()
        config.update(
            {
                "perturbation": self.perturbation,
                "a": self.a,
                "c": self.c,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "momentum": self.momentum,
            }
        )
        return config
