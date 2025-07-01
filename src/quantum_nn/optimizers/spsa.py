"""
Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

This module implements the SPSA algorithm, which is particularly well-suited
for optimizing quantum circuits on noisy hardware where gradient computation
is expensive or unreliable. SPSA estimates gradients using only two function
evaluations regardless of the number of parameters, making it highly efficient
for high-dimensional optimization problems.

The SPSA algorithm estimates gradients using simultaneous perturbations:
    ĝₖ(θ) = [f(θₖ + cₖδₖ) - f(θₖ - cₖδₖ)] / (2cₖδₖ)

where:
- θₖ are the parameters at iteration k
- cₖ is the perturbation magnitude (decreases with iteration)
- δₖ is a random perturbation vector with ±1 components
- f is the loss function

Key advantages for quantum optimization:
- Requires only 2 function evaluations per iteration (vs. 2n for finite differences)
- Robust to noise in function evaluations
- Works well with measurement noise on quantum hardware
- Asymptotically unbiased gradient estimates

The algorithm includes adaptive step sizes that decrease according to:
- aₖ = a / (k + 1)^α  (step size)
- cₖ = c / (k + 1)^γ  (perturbation size)

where α and γ are decay parameters typically set to 0.602 and 0.101 respectively.

References:
- Spall, J.C. "Introduction to Stochastic Search and Optimization" (2003)
- Spall, J.C. "Multivariate stochastic approximation using a simultaneous
  perturbation gradient approximation" IEEE Trans. Automat. Control (1992)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from quantum_nn.optimizers.quantum_aware_optimizer import QuantumAwareOptimizer


class SPSAOptimizer(QuantumAwareOptimizer):
    """
    Optimizer using Simultaneous Perturbation Stochastic Approximation.

    SPSA is particularly effective for quantum circuit optimization on noisy
    hardware, as it requires fewer function evaluations than traditional
    gradient-based methods and is robust to measurement noise. The algorithm
    estimates gradients using random simultaneous perturbations of all
    parameters, achieving convergence with only O(1) function evaluations
    per iteration instead of O(n) for finite differences.

    The optimizer implements momentum-based updates on top of SPSA gradient
    estimates for improved convergence stability.

    Key features:
    - Efficient gradient estimation with only 2 function evaluations
    - Robust to noise in quantum measurements
    - Adaptive step sizes and perturbation magnitudes
    - Momentum-based parameter updates
    - Suitable for high-dimensional parameter spaces

    Examples:
        Basic usage for quantum circuit optimization:

        >>> def noisy_circuit(params, inputs):
        ...     # Simulate noisy quantum circuit evaluation
        ...     return expectation_value_with_noise
        >>>
        >>> optimizer = SPSAOptimizer(
        ...     circuit_evaluator=noisy_circuit,
        ...     learning_rate=0.1,
        ...     a=0.01,  # step size scale
        ...     c=0.1    # perturbation scale
        ... )

        With custom decay parameters:

        >>> optimizer = SPSAOptimizer(
        ...     circuit_evaluator=noisy_circuit,
        ...     alpha=0.602,  # step size decay
        ...     gamma=0.101,  # perturbation decay
        ...     momentum=0.9
        ... )

    Note:
        The default parameters (a=0.01, c=0.1, α=0.602, γ=0.101) are based
        on theoretical guidelines but may need tuning for specific problems.
        The perturbation magnitude c should be chosen relative to the
        measurement noise level.
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
            circuit_evaluator (Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]):
                Function that evaluates the quantum circuit and returns a
                scalar loss value. Should accept parameters and optional inputs.
            perturbation (float, optional): Initial perturbation magnitude
                (deprecated, use 'c' parameter instead). Defaults to 0.01.
            a (float, optional): Scale parameter for the step size sequence.
                Controls the magnitude of parameter updates. Larger values
                lead to larger initial steps. Defaults to 0.01.
            c (float, optional): Scale parameter for the perturbation magnitude
                sequence. Controls the size of perturbations used for gradient
                estimation. Should be proportional to measurement noise level.
                Defaults to 0.1.
            alpha (float, optional): Decay parameter for step size sequence.
                Theoretical optimal value is 0.602. Controls how quickly
                step sizes decrease. Defaults to 0.602.
            gamma (float, optional): Decay parameter for perturbation magnitude
                sequence. Theoretical optimal value is 0.101. Controls how
                quickly perturbations decrease. Defaults to 0.101.
            learning_rate (Union[float, tf.keras.optimizers.schedules.LearningRateSchedule], optional):
                Overall learning rate multiplier. Defaults to 0.01.
            momentum (float, optional): Momentum coefficient for parameter
                updates. Higher values provide more smoothing but may slow
                convergence. Should be in [0, 1). Defaults to 0.9.
            name (str, optional): Name of the optimizer instance.
                Defaults to "SPSAOptimizer".
            **kwargs: Additional arguments passed to the base optimizer.

        Note:
            The step size aₖ at iteration k is computed as: aₖ = a/(k+1)^α
            The perturbation cₖ at iteration k is: cₖ = c/(k+1)^γ

            Guidelines for parameter selection:
            - a: Start with a = 0.01 and adjust based on convergence speed
            - c: Should be ~1-2 times the standard deviation of measurement noise
            - α, γ: Use theoretical values (0.602, 0.101) unless specific tuning needed
        """
        super().__init__(name=name, learning_rate=learning_rate, **kwargs)
        self.circuit_evaluator = circuit_evaluator
        self.perturbation = perturbation  # Kept for backward compatibility
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.momentum = momentum

        # Initialize optimizer state
        self._iteration = tf.Variable(0, name="iteration", dtype=tf.int64)
        self._velocity: Dict[Any, Any] = {}  # Momentum velocity

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
        lr = self._get_current_learning_rate()

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
