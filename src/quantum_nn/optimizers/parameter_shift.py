"""
Parameter-shift gradient optimizer for quantum neural networks.

This module implements the parameter-shift rule for computing gradients
of quantum circuits. The parameter-shift rule is a quantum-native method
for gradient estimation that is exact (up to statistical sampling) and
compatible with quantum hardware, unlike automatic differentiation which
requires classical simulation.

The parameter-shift rule works by evaluating the quantum circuit at shifted
parameter values and using the difference to estimate gradients. For a
parameter θ, the gradient is computed as:

    ∂⟨ψ(θ)|H|ψ(θ)⟩/∂θ = [⟨ψ(θ+s)|H|ψ(θ+s)⟩ - ⟨ψ(θ-s)|H|ψ(θ-s)⟩] / (2 sin(s))

where s is the shift amount (typically π/2) and H is the observable.

This optimizer also includes momentum-based updates similar to Adam optimizer
for improved convergence properties.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from quantum_nn.optimizers.quantum_aware_optimizer import QuantumAwareOptimizer


class ParameterShiftOptimizer(QuantumAwareOptimizer):
    """
    Optimizer using the parameter-shift rule for quantum gradients.

    This optimizer computes gradients using the parameter-shift rule,
    which is compatible with quantum hardware and avoids the need for
    backpropagation through the quantum circuit. The method is exact
    for quantum circuits with gates that have specific shift properties.

    The optimizer implements an Adam-like momentum scheme on top of the
    parameter-shift gradient computation for improved convergence.

    Key features:
    - Exact gradient computation for quantum circuits
    - Hardware-compatible (no need for classical simulation)
    - Adaptive moment estimation (similar to Adam)
    - Automatic detection of quantum vs classical parameters

    Examples:
        Basic usage with a quantum circuit:

        >>> def circuit_evaluator(params, inputs):
        ...     # Evaluate quantum circuit and return expectation values
        ...     return tf.random.normal([32])  # Example output
        >>>
        >>> optimizer = ParameterShiftOptimizer(
        ...     circuit_evaluator=circuit_evaluator,
        ...     learning_rate=0.01,
        ...     shift=np.pi/2
        ... )

    References:
        - Mitarai et al. "Quantum circuit learning" Phys. Rev. A 98, 032309 (2018)
        - Schuld et al. "Evaluating analytic gradients on quantum hardware"
          Phys. Rev. A 99, 032331 (2019)
    """

    def __init__(
        self,
        circuit_evaluator: Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor],
        shift: float = np.pi / 2,
        learning_rate: Union[
            float, tf.keras.optimizers.schedules.LearningRateSchedule
        ] = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        name: str = "ParameterShiftOptimizer",
        **kwargs,
    ):
        """
        Initialize a parameter-shift optimizer.

        Args:
            circuit_evaluator (Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]):
                Function that evaluates the quantum circuit and returns
                expectation values or loss. Should accept parameters as first
                argument and optional inputs as second argument.
            shift (float, optional): Parameter shift amount for gradient
                computation. For standard quantum gates, π/2 is optimal.
                Defaults to π/2.
            learning_rate (Union[float, tf.keras.optimizers.schedules.LearningRateSchedule], optional):
                Learning rate for parameter updates. Defaults to 0.01.
            beta_1 (float, optional): Exponential decay rate for first moment
                estimates (momentum). Should be in [0, 1). Defaults to 0.9.
            beta_2 (float, optional): Exponential decay rate for second moment
                estimates (variance). Should be in [0, 1). Defaults to 0.999.
            epsilon (float, optional): Small constant for numerical stability
                in the denominator. Defaults to 1e-7.
            name (str, optional): Name of the optimizer instance.
                Defaults to "ParameterShiftOptimizer".
            **kwargs: Additional arguments passed to the base optimizer.

        Note:
            The circuit_evaluator function should be designed to work with
            batched inputs and return scalar or vector outputs suitable
            for gradient computation.
        """
        super().__init__(name=name, learning_rate=learning_rate, **kwargs)
        self.circuit_evaluator = circuit_evaluator
        self.shift = shift
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # Initialize optimizer state
        self._iteration = tf.Variable(0, name="iteration", dtype=tf.int64)
        self._m: Dict[Any, Any] = {}  # First moment
        self._v: Dict[Any, Any] = {}  # Second moment

    def _create_slots(self, var_list: List[tf.Variable]):
        """
        Create optimizer state variables for momentum tracking.

        Creates first and second moment variables for each trainable variable
        to implement the Adam-like momentum scheme. These variables are used
        to store running averages of gradients and squared gradients.

        Args:
            var_list (List[tf.Variable]): List of variables to create slots for.
                Each variable will have corresponding momentum variables created.

        Note:
            This method is called automatically when needed and creates
            TensorFlow variables that persist across optimization steps.
        """
        for var in var_list:
            self._m[var.ref()] = tf.Variable(
                tf.zeros_like(var), name=f"{var.name}/m", trainable=False
            )
            self._v[var.ref()] = tf.Variable(
                tf.zeros_like(var), name=f"{var.name}/v", trainable=False
            )

    @tf.function
    def _parameter_shift_gradient(
        self, params: tf.Tensor, inputs: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Compute gradients using the parameter-shift rule.

        The parameter-shift rule computes exact gradients for quantum circuits
        by evaluating the circuit at shifted parameter values. For each parameter,
        the gradient is computed as the difference between function evaluations
        at θ + shift and θ - shift, scaled by the trigonometric factor.

        Args:
            params (tf.Tensor): Quantum circuit parameters as a 1D tensor.
                Each element corresponds to a gate parameter in the circuit.
            inputs (Optional[tf.Tensor], optional): Input data for the quantum
                circuit, if applicable. Defaults to None.

        Returns:
            tf.Tensor: Gradient tensor with the same shape as params.
                Each element is the partial derivative with respect to the
                corresponding parameter.

        Note:
            This method uses TensorFlow's @tf.function decorator for performance.
            The shift amount and circuit evaluator are used to compute exact
            gradients that are compatible with quantum hardware execution.

        Mathematical Details:
            For parameter θᵢ, the gradient is:
            ∂f/∂θᵢ = [f(θ + s·eᵢ) - f(θ - s·eᵢ)] / (2·sin(s))
            where s is the shift amount and eᵢ is the i-th unit vector.
        """
        gradients = tf.TensorArray(tf.float32, size=params.shape[0])

        for i in range(params.shape[0]):
            # Create shifted parameters
            shift_forward = tf.tensor_scatter_nd_update(
                params, [[i]], [params[i] + self.shift]
            )
            shift_backward = tf.tensor_scatter_nd_update(
                params, [[i]], [params[i] - self.shift]
            )

            # Evaluate circuit with shifted parameters
            forward = self.circuit_evaluator(shift_forward, inputs)
            backward = self.circuit_evaluator(shift_backward, inputs)

            # Compute gradient using parameter-shift rule
            gradient = tf.reduce_mean(forward - backward) / (2 * tf.sin(self.shift))
            gradients = gradients.write(i, gradient)

        return gradients.stack()

    def apply_gradients(
        self,
        grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]],
        name: Optional[str] = None,
        **kwargs,
    ) -> tf.Operation:
        """
        Apply gradients to variables using parameter-shift rule and Adam-like momentum.

        This method implements the core optimization logic, combining parameter-shift
        gradient computation with Adam-style momentum updates. For variables identified
        as quantum parameters (by name prefix), gradients are computed using the
        parameter-shift rule. For classical parameters, standard gradients are used.

        Args:
            grads_and_vars (List[Tuple[tf.Tensor, tf.Variable]]): List of
                (gradient, variable) pairs. For quantum variables, the provided
                gradients are replaced with parameter-shift computed gradients.
            name (Optional[str], optional): Name for the operation.
                Defaults to None.
            **kwargs: Additional keyword arguments including:
                - inputs: Input data for quantum circuit evaluation (required
                  for parameter-shift gradient computation)

        Returns:
            tf.Operation: A TensorFlow operation that applies the updates.

        Note:
            Variables with names starting with "quantum" are treated as quantum
            circuit parameters and use parameter-shift gradient computation.
            All other variables use the provided gradients with Adam momentum.
        """
        # Create slots if they don't exist
        var_list = [var for _, var in grads_and_vars]
        if not self._m:
            self._create_slots(var_list)

        # Increment iteration counter
        self._iteration.assign_add(1)

        # Compute bias correction terms
        lr = self._get_current_learning_rate()
        beta_1_t = tf.pow(self.beta_1, tf.cast(self._iteration, tf.float32))
        beta_2_t = tf.pow(self.beta_2, tf.cast(self._iteration, tf.float32))
        alpha = lr * tf.sqrt(1 - beta_2_t) / (1 - beta_1_t)

        # Apply updates to each variable
        for grad, var in grads_and_vars:
            if grad is None:
                continue

            # For quantum layers, compute gradients using parameter-shift rule
            if var.name.startswith("quantum"):
                inputs = kwargs.get("inputs")
                grad = self._parameter_shift_gradient(var, inputs)

            # Update moments
            m = self._m[var.ref()]
            v = self._v[var.ref()]

            m.assign(self.beta_1 * m + (1 - self.beta_1) * grad)
            v.assign(self.beta_2 * v + (1 - self.beta_2) * tf.square(grad))

            # Apply update
            var.assign_sub(alpha * m / (tf.sqrt(v) + self.epsilon))

        return tf.no_op(name=name)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the optimizer.

        Returns the optimizer's configuration including all hyperparameters
        needed to recreate the optimizer instance. This includes both the
        base class parameters and optimizer-specific parameters.

        Returns:
            Dict[str, Any]: Configuration dictionary containing:
                - shift: Parameter shift amount used for gradient computation
                - beta_1: First moment decay rate
                - beta_2: Second moment decay rate
                - epsilon: Numerical stability constant
                - Plus all base class configuration parameters

        Example:
            >>> optimizer = ParameterShiftOptimizer(circuit_evaluator=my_circuit)
            >>> config = optimizer.get_config()
            >>> new_optimizer = ParameterShiftOptimizer.from_config(config)
        """
        config = super().get_config()
        config.update(
            {
                "shift": self.shift,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
            }
        )
        return config
