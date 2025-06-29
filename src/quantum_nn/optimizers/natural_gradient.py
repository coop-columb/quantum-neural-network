"""
Quantum natural gradient optimizer.

This module implements the quantum natural gradient method,
which takes into account the geometry of the quantum state space
for more efficient optimization.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from quantum_nn.optimizers.quantum_aware_optimizer import QuantumAwareOptimizer


@tf.keras.utils.register_keras_serializable(package="QuantumNN")


class QuantumNaturalGradient(QuantumAwareOptimizer):
    """
    Optimizer using quantum natural gradient for quantum neural networks.

    This optimizer computes the quantum Fisher information matrix to
    account for the Riemannian geometry of the quantum state space,
    leading to more efficient optimization.
    """

    def __init__(
        self,
        circuit_evaluator: Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor],
        fidelity_estimator: Optional[
            Callable[[tf.Tensor, tf.Tensor, Optional[tf.Tensor]], tf.Tensor]
        ] = None,
        damping: float = 0.01,
        learning_rate: Union[
            float, tf.keras.optimizers.schedules.LearningRateSchedule
        ] = 0.01,
        name: str = "QuantumNaturalGradient",
        **kwargs,
    ):
        """
        Initialize a quantum natural gradient optimizer.

        Args:
            circuit_evaluator: Function to evaluate the quantum circuit
            fidelity_estimator: Function to estimate quantum fidelity between states
            damping: Damping factor for the Fisher information matrix
            learning_rate: Initial learning rate or schedule
            name: Name of the optimizer
            **kwargs: Additional arguments to pass to the optimizer
        """
        super().__init__(name=name, learning_rate=learning_rate, **kwargs)
        self.circuit_evaluator = circuit_evaluator
        self.fidelity_estimator = fidelity_estimator or self._default_fidelity_estimator
        self.damping = damping

    def _default_fidelity_estimator(
        self, params1: tf.Tensor, params2: tf.Tensor, inputs: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Default fidelity estimator using circuit overlap.

        Args:
            params1: First set of circuit parameters
            params2: Second set of circuit parameters
            inputs: Input data (optional)

        Returns:
            Estimated fidelity between the quantum states
        """
        # Evaluate circuits with both parameter sets
        state1 = self.circuit_evaluator(params1, inputs)
        state2 = self.circuit_evaluator(params2, inputs)

        # Compute fidelity as squared overlap
        fidelity = tf.reduce_sum(state1 * state2, axis=-1) ** 2
        return fidelity

    @tf.function
    def _compute_fisher_matrix(
        self, params: tf.Tensor, inputs: Optional[tf.Tensor] = None, eps: float = 0.01
    ) -> tf.Tensor:
        """
        Compute the quantum Fisher information matrix.

        Args:
            params: Quantum circuit parameters
            inputs: Input data (optional)
            eps: Small perturbation for finite difference

        Returns:
            Quantum Fisher information matrix
        """
        n_params = params.shape[0]
        fisher_matrix = tf.TensorArray(tf.float32, size=n_params * n_params)

        # Compute Fisher matrix elements
        idx = 0
        for i in range(n_params):
            for j in range(n_params):
                # Perturb parameters
                params_i_plus = tf.tensor_scatter_nd_update(
                    params, [[i]], [params[i] + eps]
                )
                params_j_plus = tf.tensor_scatter_nd_update(
                    params, [[j]], [params[j] + eps]
                )

                # Compute fidelity and derivatives
                fid_i = self.fidelity_estimator(params, params_i_plus, inputs)
                fid_j = self.fidelity_estimator(params, params_j_plus, inputs)
                fid_ij = self.fidelity_estimator(params_i_plus, params_j_plus, inputs)

                # Approximate second derivative of fidelity
                fisher_ij = (2 - 2 * fid_ij - (2 - 2 * fid_i) - (2 - 2 * fid_j)) / (
                    eps**2
                )
                fisher_matrix = fisher_matrix.write(idx, fisher_ij)
                idx += 1

        # Reshape to matrix form
        fisher_matrix = fisher_matrix.stack()
        fisher_matrix = tf.reshape(fisher_matrix, [n_params, n_params])

        # Add damping for numerical stability
        fisher_matrix += tf.eye(n_params) * self.damping

        return fisher_matrix

    @tf.function
    def _natural_gradient(
        self, params: tf.Tensor, grad: tf.Tensor, inputs: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Compute natural gradient using the Fisher information matrix.

        Args:
            params: Quantum circuit parameters
            grad: Gradient with respect to parameters
            inputs: Input data (optional)

        Returns:
            Natural gradient
        """
        # Compute Fisher information matrix
        fisher_matrix = self._compute_fisher_matrix(params, inputs)

        # Solve the linear system F⁻¹ ∇J
        natural_grad = tf.linalg.solve(fisher_matrix, grad)

        return natural_grad

    def apply_gradients(
        self,
        grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]],
        name: Optional[str] = None,
        **kwargs,
    ) -> tf.Operation:
        """
        Apply natural gradients to variables.

        Args:
            grads_and_vars: List of (gradient, variable) pairs
            name: Name of the operation
            **kwargs: Additional arguments

        Returns:
            An operation that applies the gradients
        """
        lr = self._get_hyper("learning_rate", tf.float32)

        # Apply updates to each variable
        for grad, var in grads_and_vars:
            if grad is None:
                continue

            # For quantum layers, compute natural gradient
            if var.name.startswith("quantum"):
                inputs = kwargs.get("inputs")
                natural_grad = self._natural_gradient(var, grad, inputs)
                var.assign_sub(lr * natural_grad)
            else:
                # For classical layers, use standard gradient
                var.assign_sub(lr * grad)

        return tf.no_op(name=name)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the optimizer."""
        config = super().get_config()
        config.update(
            {
                "damping": self.damping,
            }
        )
        return config
