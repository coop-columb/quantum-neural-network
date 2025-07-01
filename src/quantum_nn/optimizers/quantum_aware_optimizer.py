"""
Base class for quantum-aware optimizers.

This module provides the foundation for optimization algorithms
specifically designed for quantum neural networks. It extends TensorFlow's
optimizer interface to support quantum circuit parameter optimization.

The quantum-aware optimizers in this module are designed to handle the
unique challenges of quantum machine learning, including:
- Parameter-shift rule computation for gradient estimation
- Quantum Fisher information matrix calculations
- Noise-aware optimization techniques
- Integration with quantum circuit evaluators
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf


class QuantumAwareOptimizer(tf.keras.optimizers.Optimizer):
    """
    Base class for quantum-aware optimizers.

    This class extends TensorFlow's optimizer interface to provide
    specialized optimization techniques for quantum neural networks.
    Subclasses implement specific quantum-aware optimization algorithms
    such as parameter-shift rule, quantum natural gradient, and SPSA.

    The key features of quantum-aware optimizers include:
    - Support for quantum circuit parameter optimization
    - Integration with quantum circuit evaluators
    - Handling of quantum-specific gradient computation methods
    - Compatibility with TensorFlow's optimization ecosystem

    Examples:
        Creating a custom quantum-aware optimizer:

        >>> class MyQuantumOptimizer(QuantumAwareOptimizer):
        ...     def apply_gradients(self, grads_and_vars, **kwargs):
        ...         # Custom gradient application logic
        ...         pass

    Note:
        This is an abstract base class. Subclasses must implement the
        `apply_gradients` method to define how gradients are applied
        to quantum circuit parameters.
    """

    def __init__(
        self,
        name: str,
        learning_rate: Union[
            float, tf.keras.optimizers.schedules.LearningRateSchedule
        ] = 0.01,
        **kwargs,
    ):
        """
        Initialize a quantum-aware optimizer.

        Args:
            name (str): Name of the optimizer instance. Used for TensorFlow
                variable naming and debugging purposes.
            learning_rate (Union[float, tf.keras.optimizers.schedules.LearningRateSchedule], optional):
                Initial learning rate for parameter updates. Can be a constant
                float value or a learning rate schedule that changes over time.
                Defaults to 0.01.
            **kwargs: Additional keyword arguments passed to the parent
                TensorFlow optimizer class. May include clipnorm, clipvalue,
                global_clipnorm, etc.

        Note:
            The learning rate is automatically converted to a TensorFlow
            variable or preserved as a schedule object for efficient
            computation during optimization.
        """
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)

    def apply_gradients(
        self,
        grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]],
        name: Optional[str] = None,
        **kwargs,
    ) -> tf.Operation:
        """
        Apply gradients to variables using quantum-aware optimization.

        This is an abstract method that must be implemented by subclasses
        to define how gradients are applied to quantum circuit parameters.
        Different quantum-aware optimizers implement different strategies
        for gradient computation and application.

        Args:
            grads_and_vars (List[Tuple[tf.Tensor, tf.Variable]]): List of
                (gradient, variable) pairs where gradients are applied to
                the corresponding variables. Gradients may be None for
                variables that don't require updates.
            name (Optional[str], optional): Name for the operation in the
                TensorFlow graph. Used for debugging and visualization.
                Defaults to None.
            **kwargs: Additional keyword arguments that may be used by
                specific optimizer implementations. Common arguments include:
                - inputs: Input data for quantum circuit evaluation
                - global_step: Current training step for adaptive methods

        Returns:
            tf.Operation: A TensorFlow operation that applies the gradients
                when executed. This operation should update all variables
                according to the optimizer's strategy.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Note:
            Subclasses should handle quantum-specific gradient computation
            methods like parameter-shift rule or finite difference schemes
            when applying gradients to quantum circuit parameters.
        """
        raise NotImplementedError("Subclasses must implement apply_gradients")

    def get_config(self) -> Dict:
        """
        Return the configuration of the optimizer.

        Returns a dictionary containing the optimizer's configuration
        parameters that can be used to recreate the optimizer instance.
        This is used for serialization and model saving/loading.

        Returns:
            Dict: Configuration dictionary containing the optimizer's
                parameters including:
                - learning_rate: Current learning rate value or serialized schedule
                - Additional parameters specific to the optimizer subclass

        Note:
            The returned configuration can be used with the optimizer's
            `from_config` class method to recreate an identical optimizer
            instance.
        """
        return super().get_config()
