"""
Parameter-shift gradient optimizer for quantum neural networks.

This module implements the parameter-shift rule for computing
gradients of quantum circuits, which is more suitable for
quantum hardware than automatic differentiation.
"""
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import numpy as np
import tensorflow as tf

from quantum_nn.optimizers.quantum_aware_optimizer import QuantumAwareOptimizer


class ParameterShiftOptimizer(QuantumAwareOptimizer):
    """
    Optimizer using the parameter-shift rule for quantum gradients.
    
    This optimizer computes gradients using the parameter-shift rule,
    which is compatible with quantum hardware and avoids the need for
    backpropagation through the quantum circuit.
    """
    
    def __init__(
        self,
        circuit_evaluator: Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor],
        shift: float = np.pi/2,
        learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        name: str = "ParameterShiftOptimizer",
        **kwargs
    ):
        """
        Initialize a parameter-shift optimizer.
        
        Args:
            circuit_evaluator: Function to evaluate the quantum circuit
            shift: Parameter shift amount (typically π/2)
            learning_rate: Initial learning rate or schedule
            beta_1: Exponential decay rate for first moment
            beta_2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            name: Name of the optimizer
            **kwargs: Additional arguments to pass to the optimizer
        """
        super().__init__(name=name, learning_rate=learning_rate, **kwargs)
        self.circuit_evaluator = circuit_evaluator
        self.shift = shift
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        # Initialize optimizer state
        self._iteration = tf.Variable(0, name="iteration", dtype=tf.int64)
        self._m = {}  # First moment
        self._v = {}  # Second moment
    
    def _create_slots(self, var_list: List[tf.Variable]):
        """Create optimizer state variables."""
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
        
        Args:
            params: Quantum circuit parameters
            inputs: Input data (optional)
            
        Returns:
            Gradients with respect to the parameters
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
        **kwargs
    ) -> tf.Operation:
        """
        Apply gradients to variables using parameter-shift rule.
        
        Args:
            grads_and_vars: List of (gradient, variable) pairs
            name: Name of the operation
            **kwargs: Additional arguments
            
        Returns:
            An operation that applies the gradients
        """
        # Create slots if they don't exist
        var_list = [var for _, var in grads_and_vars]
        if not self._m:
            self._create_slots(var_list)
        
        # Increment iteration counter
        self._iteration.assign_add(1)
        
        # Compute bias correction terms
        lr = self._get_hyper("learning_rate", tf.float32)
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
        """Return the configuration of the optimizer."""
        config = super().get_config()
        config.update({
            "shift": self.shift,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
        })
        return config
