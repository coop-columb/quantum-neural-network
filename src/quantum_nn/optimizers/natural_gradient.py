"""
Quantum natural gradient optimizer.

This module implements the quantum natural gradient method, which
takes into account the geometry of the quantum state space for more
efficient optimization. The quantum natural gradient uses the quantum
Fisher information matrix to define a Riemannian metric on the space
of quantum states, leading to optimization that follows the steepest
descent direction in this curved space.

The quantum natural gradient direction is computed as:
    θ_{t+1} = θ_t - η F^{-1} ∇L

where F is the quantum Fisher information matrix, η is the learning rate,
and ∇L is the standard gradient. The Fisher information matrix encodes
the geometric structure of the quantum state space and provides better
convergence properties compared to standard gradient descent.

Key advantages:
- Accounts for the intrinsic geometry of quantum state space
- Often requires fewer optimization steps than standard methods
- Particularly effective for variational quantum algorithms
- Provides natural units for parameter updates

References:
- Stokes et al. "Quantum Natural Gradient" Quantum 4, 269 (2020)
- Yamamoto "On the natural gradient for variational quantum eigensolver"
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from quantum_nn.optimizers.quantum_aware_optimizer import QuantumAwareOptimizer


class QuantumNaturalGradient(QuantumAwareOptimizer):
    """
    Optimizer using quantum natural gradient for quantum neural networks.

    This optimizer computes the quantum Fisher information matrix to
    account for the Riemannian geometry of the quantum state space,
    leading to more efficient optimization. The natural gradient direction
    is computed by inverting the Fisher information matrix and applying
    it to the standard gradient.

    The quantum Fisher information matrix F_ij is defined as:
    F_ij = Re[⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩]

    where |ψ⟩ is the quantum state and ∂_i denotes partial derivative
    with respect to the i-th parameter.

    Key features:
    - Geometric optimization using Fisher information
    - Configurable fidelity estimation methods
    - Damping for numerical stability
    - Automatic fallback to standard gradients for classical parameters

    Examples:
        Basic usage with default fidelity estimation:

        >>> def my_circuit(params, inputs):
        ...     # Quantum circuit evaluation
        ...     return quantum_expectation_values
        >>>
        >>> optimizer = QuantumNaturalGradient(
        ...     circuit_evaluator=my_circuit,
        ...     learning_rate=0.01,
        ...     damping=0.001
        ... )

        With custom fidelity estimator:

        >>> def custom_fidelity(params1, params2, inputs):
        ...     # Custom fidelity computation
        ...     return fidelity_value
        >>>
        >>> optimizer = QuantumNaturalGradient(
        ...     circuit_evaluator=my_circuit,
        ...     fidelity_estimator=custom_fidelity
        ... )

    Note:
        The Fisher information matrix computation scales as O(n²) where n
        is the number of parameters, making this method most suitable for
        circuits with moderate numbers of parameters (typically < 100).
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
            circuit_evaluator (Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]):
                Function that evaluates the quantum circuit and returns
                expectation values or quantum states. Should accept parameters
                as first argument and optional inputs as second argument.
            fidelity_estimator (Optional[Callable], optional): Function to
                estimate quantum fidelity between two quantum states produced
                by different parameter sets. If None, uses the default overlap-based
                estimator. Should accept (params1, params2, inputs) and return
                scalar fidelity values. Defaults to None.
            damping (float, optional): Damping factor added to the diagonal
                of the Fisher information matrix for numerical stability.
                Larger values make the optimizer more conservative but more
                stable. Defaults to 0.01.
            learning_rate (Union[float, tf.keras.optimizers.schedules.LearningRateSchedule], optional):
                Learning rate for parameter updates. Defaults to 0.01.
            name (str, optional): Name of the optimizer instance.
                Defaults to "QuantumNaturalGradient".
            **kwargs: Additional arguments passed to the base optimizer.

        Note:
            The circuit_evaluator should return consistent outputs for the
            same parameters to ensure accurate Fisher information computation.
            The fidelity_estimator is crucial for the Fisher matrix quality.
        """
        super().__init__(name=name, learning_rate=learning_rate, **kwargs)
        self.circuit_evaluator = circuit_evaluator
        self.fidelity_estimator = fidelity_estimator or self._default_fidelity_estimator
        self.damping = damping

    def _default_fidelity_estimator(
        self, params1: tf.Tensor, params2: tf.Tensor, inputs: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Default fidelity estimator using circuit state overlap.

        Computes fidelity between quantum states generated by two different
        parameter sets using the squared overlap |⟨ψ(θ₁)|ψ(θ₂)⟩|². This is
        a reasonable default for many quantum circuits but may not be optimal
        for all use cases.

        Args:
            params1 (tf.Tensor): First set of circuit parameters.
            params2 (tf.Tensor): Second set of circuit parameters.
            inputs (Optional[tf.Tensor], optional): Input data for the
                quantum circuit, if applicable. Defaults to None.

        Returns:
            tf.Tensor: Estimated fidelity between the quantum states,
                typically a scalar value between 0 and 1.

        Note:
            This implementation assumes the circuit_evaluator returns
            state vectors or amplitude arrays that can be used to compute
            overlaps. For expectation value-based circuits, a custom
            fidelity estimator should be provided.

        Mathematical Details:
            For pure states |ψ₁⟩ and |ψ₂⟩, the fidelity is:
            F(ψ₁, ψ₂) = |⟨ψ₁|ψ₂⟩|²
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

        The quantum Fisher information matrix encodes the geometric structure
        of the quantum state space. It is computed using finite differences
        of the fidelity function, which approximates the metric tensor of
        the parameter space.

        Args:
            params (tf.Tensor): Quantum circuit parameters as a 1D tensor.
            inputs (Optional[tf.Tensor], optional): Input data for quantum
                circuit evaluation. Defaults to None.
            eps (float, optional): Small perturbation for finite difference
                computation. Larger values give more stable but less accurate
                estimates. Defaults to 0.01.

        Returns:
            tf.Tensor: Fisher information matrix of shape [n_params, n_params]
                where n_params is the number of circuit parameters. The matrix
                is symmetric and positive semi-definite.

        Mathematical Details:
            The Fisher information matrix element F_ij is approximated as:
            F_ij ≈ (2 - 2*fid_ij - (2 - 2*fid_i) - (2 - 2*fid_j)) / ε²

            where:
            - fid_i = F(θ, θ + ε·e_i)
            - fid_j = F(θ, θ + ε·e_j)
            - fid_ij = F(θ + ε·e_i, θ + ε·e_j)
            - e_i is the i-th unit vector

        Note:
            This method uses TensorFlow's @tf.function decorator for performance
            but scales as O(n²) with the number of parameters. For large
            parameter spaces, consider approximation methods.
        """
        n_params = params.shape[0]
        fisher_matrix = tf.TensorArray(tf.float32, size=n_params * n_params)

        # Compute Fisher matrix elements using finite differences
        idx = 0
        for i in range(n_params):
            for j in range(n_params):
                # Perturb parameters in both directions
                params_i_plus = tf.tensor_scatter_nd_update(
                    params, [[i]], [params[i] + eps]
                )
                params_j_plus = tf.tensor_scatter_nd_update(
                    params, [[j]], [params[j] + eps]
                )

                # Compute fidelities for Fisher matrix approximation
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
        lr = self._get_current_learning_rate()

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
