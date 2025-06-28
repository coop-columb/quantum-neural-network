"""
Base quantum model class using PennyLane.

This module provides the base class for quantum neural network models
that integrates with TensorFlow Keras.
"""
from typing import List, Union, Optional, Dict, Any
import tensorflow as tf

from ..layers.quantum_layer import QuantumLayer


class QuantumModel(tf.keras.Model):
    """
    Base class for quantum neural network models.
    
    This class extends tf.keras.Model to provide quantum-specific
    functionality while maintaining compatibility with standard
    TensorFlow/Keras workflows.
    """
    
    def __init__(self, layers: Optional[List[Union[tf.keras.layers.Layer, Dict]]] = None, **kwargs):
        """
        Initialize the quantum model.
        
        Args:
            layers: List of layers or layer configurations
            **kwargs: Additional model arguments
        """
        super().__init__(**kwargs)
        
        self.model_layers = []
        
        if layers:
            for layer in layers:
                if isinstance(layer, dict):
                    # Create layer from configuration
                    self.model_layers.append(self._create_layer_from_config(layer))
                else:
                    # Add existing layer
                    self.model_layers.append(layer)
    
    def _create_layer_from_config(self, config: Dict) -> tf.keras.layers.Layer:
        """Create a layer from configuration dictionary."""
        layer_type = config.get('type', 'dense')
        
        if layer_type == 'quantum':
            return QuantumLayer(
                n_qubits=config.get('n_qubits', 4),
                measurement_type=config.get('measurement_type', 'expectation')
            )
        elif layer_type == 'dense':
            return tf.keras.layers.Dense(
                units=config.get('units', 32),
                activation=config.get('activation', 'relu')
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    def call(self, inputs, training=None):
        """Forward pass through the model."""
        x = inputs
        
        for layer in self.model_layers:
            x = layer(x, training=training)
        
        return x
    
    def compile(self, optimizer='adam', loss=None, metrics=None, **kwargs):
        """Compile the quantum model."""
        # Set default loss if not provided
        if loss is None:
            loss = 'mse'
        
        # Set default metrics if not provided
        if metrics is None:
            metrics = ['mae']
        
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
    
    def summary(self):
        """Print model summary."""
        print(f"Quantum Model Summary:")
        print(f"Number of layers: {len(self.model_layers)}")
        
        for i, layer in enumerate(self.model_layers):
            if isinstance(layer, QuantumLayer):
                print(f"Layer {i}: QuantumLayer ({layer.n_qubits} qubits, {layer.measurement_type})")
            else:
                print(f"Layer {i}: {layer.__class__.__name__}")
        
        # Try to build model for detailed summary
        try:
            super().summary()
        except:
            print("(Detailed summary requires model to be built)")
