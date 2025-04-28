"""
Base class for quantum-aware optimizers.

This module provides the foundation for optimization algorithms
specifically designed for quantum neural networks.
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf


class QuantumAwareOptimizer(tf.keras.optimizers.Optimizer):
    """
    Base class for quantum-aware optimizers.
    
    This class extends TensorFlow's optimizer interface to provide
    specialized optimization techniques for quantum neural networks.
    """
    
    def __init__(
        self,
        name: str,
        learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 0.01,
        **kwargs
    ):
        """
        Initialize a quantum-aware optimizer.
        
        Args:
            name: Name of the optimizer
            learning_rate: Initial learning rate or schedule
            **kwargs: Additional arguments to pass to the optimizer
        """
        super().__init__(name=name, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
    
    def _build_learning_rate(
        self, learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]
    ) -> Union[tf.Variable, tf.keras.optimizers.schedules.LearningRateSchedule]:
        """Build the learning rate."""
        if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            return learning_rate
        else:
            return tf.Variable(float(learning_rate), name="learning_rate")
    
    def apply_gradients(
        self,
        grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]],
        name: Optional[str] = None,
        **kwargs
    ) -> tf.Operation:
        """Apply gradients to variables."""
        raise NotImplementedError("Subclasses must implement apply_gradients")
    
    def get_config(self) -> Dict:
        """Return the configuration of the optimizer."""
        config = super().get_config()
        
        if isinstance(self._learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            config.update({
                "learning_rate": tf.keras.optimizers.schedules.serialize(self._learning_rate)
            })
        else:
            config.update({
                "learning_rate": float(self._learning_rate.numpy())
            })
        
        return config
    
    @property
    def learning_rate(self) -> Union[tf.Variable, tf.keras.optimizers.schedules.LearningRateSchedule]:
        """Get the current learning rate."""
        if isinstance(self._learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            return self._learning_rate
        else:
            return self._learning_rate.value()
    
    @learning_rate.setter
    def learning_rate(self, learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]):
        """Set the learning rate."""
        if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            self._learning_rate = learning_rate
        else:
            self._learning_rate.assign(learning_rate)
