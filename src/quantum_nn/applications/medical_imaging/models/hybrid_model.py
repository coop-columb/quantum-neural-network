"""
Hybrid classical-quantum models for medical imaging.

This module implements models that combine classical neural networks
with quantum components for enhanced medical image analysis.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

import numpy as np
import tensorflow as tf

from quantum_nn.models import QuantumModel
from quantum_nn.layers import QuantumLayer

logger = logging.getLogger(__name__)


class MedicalHybridModel(tf.keras.Model):
    """
    Hybrid classical-quantum model for medical imaging.
    
    This model combines classical convolutional layers for feature extraction
    with quantum layers for enhanced pattern recognition and classification.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_classes: int = 2,
        n_qubits: int = 8,
        n_quantum_layers: int = 2,
        classical_backbone: str = "mobilenet",
        quantum_position: str = "middle",
        model_size: str = "medium",
        use_pretrained: bool = True,
        **kwargs
    ):
        """
        Initialize hybrid medical model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            n_classes: Number of output classes
            n_qubits: Number of qubits in quantum layers
            n_quantum_layers: Number of quantum circuit layers (stored for reference)
            classical_backbone: Classical backbone architecture
            quantum_position: Where to insert quantum layers ('early', 'middle', 'late')
            model_size: Model size ('small', 'medium', 'large')
            use_pretrained: Whether to use pretrained weights
            **kwargs: Additional arguments
        """
        # Initialize parent class first
        super().__init__(**kwargs)
        
        self.input_shape_custom = input_shape
        self.n_classes = n_classes
        self.n_qubits = n_qubits
        self.n_quantum_layers = n_quantum_layers
        self.classical_backbone = classical_backbone
        self.quantum_position = quantum_position
        self.model_size = model_size
        self.use_pretrained = use_pretrained
        
        # Build the model
        self._build_hybrid_model()
        
        # Initialize medical-specific components
        self._initialize_medical_metrics()
        
        logger.info(
            f"Initialized MedicalHybridModel with {classical_backbone} backbone, "
            f"{n_qubits} qubits, quantum at {quantum_position} position"
        )
    
    def _build_hybrid_model(self):
        """Build the hybrid classical-quantum architecture."""
        inputs = tf.keras.layers.Input(shape=self.input_shape_custom)
        
        # Build based on model size
        if self.model_size == "small":
            x = self._build_small_model(inputs)
        elif self.model_size == "medium":
            x = self._build_medium_model(inputs)
        else:  # large
            x = self._build_large_model(inputs)
        
        # Output layer
        if self.n_classes == 2:
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)
        
        # Create internal functional model
        self.internal_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def call(self, inputs, training=None):
        """Forward pass using the internal model."""
        return self.internal_model(inputs, training=training)
    
    def _build_small_model(self, inputs):
        """Build small hybrid model architecture."""
        # Initial convolution
        x = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        
        # Second convolution
        x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        
        # Flatten for quantum processing
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        
        # Quantum layer with correct parameters
        quantum_layer = QuantumLayer(
            n_qubits=self.n_qubits,
            measurement_type="expectation"
        )
        x = quantum_layer(x)
        
        # Final dense layer
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        return x
    
    def _build_medium_model(self, inputs):
        """Build medium hybrid model architecture."""
        # Use pretrained MobileNet if requested
        if self.use_pretrained and self.classical_backbone == "mobilenet":
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=self.input_shape_custom,
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False  # Freeze initially
            
            x = base_model(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        else:
            # Custom CNN backbone
            x = self._build_custom_cnn(inputs)
        
        # Intermediate processing
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Quantum layers with correct parameters
        quantum_layer1 = QuantumLayer(
            n_qubits=self.n_qubits,
            measurement_type="expectation"
        )
        x = quantum_layer1(x)
        
        # Additional classical processing
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Second quantum layer
        quantum_layer2 = QuantumLayer(
            n_qubits=self.n_qubits // 2,
            measurement_type="expectation"
        )
        x = quantum_layer2(x)
        
        # Final processing
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        return x
    
    def _build_large_model(self, inputs):
        """Build large hybrid model architecture."""
        # Use pretrained EfficientNet for large model
        if self.use_pretrained:
            base_model = tf.keras.applications.EfficientNetB0(
                input_shape=self.input_shape_custom,
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            x = base_model(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        else:
            # Deep custom CNN
            x = self._build_deep_custom_cnn(inputs)
        
        # Multiple quantum-classical blocks
        for i in range(3):
            # Classical block
            x = tf.keras.layers.Dense(512 // (2**i), activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            # Quantum block with correct parameters
            quantum_layer = QuantumLayer(
                n_qubits=self.n_qubits - i*2,
                measurement_type="expectation"
            )
            x = quantum_layer(x)
        
        # Final processing
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        return x
    
    def _build_custom_cnn(self, inputs):
        """Build custom CNN backbone."""
        x = inputs
        
        # Convolutional blocks
        for filters in [32, 64, 128]:
            x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        return x
    
    def _build_deep_custom_cnn(self, inputs):
        """Build deep custom CNN backbone."""
        x = inputs
        
        # Deep convolutional blocks with residual connections
        for i, filters in enumerate([64, 128, 256, 512]):
            # Residual block
            shortcut = x
            
            x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
            # Match dimensions for residual connection
            if i > 0:
                shortcut = tf.keras.layers.Conv2D(filters, 1)(shortcut)
                shortcut = tf.keras.layers.BatchNormalization()(shortcut)
            
            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        return x
    
    def _initialize_medical_metrics(self):
        """Initialize medical-specific metrics."""
        self.medical_metrics = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            # Note: F1Score removed due to shape compatibility issues with 1D binary labels
            # For multiclass, F1Score can be added back
        ]
    
    def compile_for_medical_imaging(
        self,
        learning_rate: float = 0.001,
        loss: Optional[str] = None,
        optimizer: Optional[str] = None
    ):
        """
        Compile model with medical imaging specific settings.
        
        Args:
            learning_rate: Learning rate for optimizer
            loss: Loss function (defaults based on n_classes)
            optimizer: Optimizer name
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        if loss is None:
            loss = 'binary_crossentropy' if self.n_classes == 2 else 'categorical_crossentropy'
        
        # For multiclass, add F1Score back
        metrics = self.medical_metrics.copy()
        if self.n_classes > 2:
            metrics.append(tf.keras.metrics.F1Score(average='weighted', name='f1_score'))
        
        self.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Hybrid model compiled for medical imaging with {self.model_size} architecture")
    
    def unfreeze_backbone(self, learning_rate: float = 0.0001):
        """
        Unfreeze the classical backbone for fine-tuning.
        
        Args:
            learning_rate: Learning rate for fine-tuning
        """
        # Find and unfreeze the backbone
        for layer in self.internal_model.layers:
            if isinstance(layer, tf.keras.Model):  # Pretrained model
                layer.trainable = True
                # Keep batch norm layers frozen
                for inner_layer in layer.layers:
                    if isinstance(inner_layer, tf.keras.layers.BatchNormalization):
                        inner_layer.trainable = False
        
        # Recompile with lower learning rate
        self.compile_for_medical_imaging(learning_rate=learning_rate)
        
        logger.info("Backbone unfrozen for fine-tuning")
    
    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_custom,
            'n_classes': self.n_classes,
            'n_qubits': self.n_qubits,
            'n_quantum_layers': self.n_quantum_layers,
            'classical_backbone': self.classical_backbone,
            'quantum_position': self.quantum_position,
            'model_size': self.model_size,
            'use_pretrained': self.use_pretrained,
        })
        return config


def create_hybrid_medical_model(
    input_shape: Tuple[int, int, int],
    n_classes: int = 2,
    model_size: str = "medium",
    use_pretrained: bool = True,
    **kwargs
) -> MedicalHybridModel:
    """
    Factory function to create a hybrid medical model.
    
    Args:
        input_shape: Input image shape
        n_classes: Number of output classes
        model_size: Model size ('small', 'medium', 'large')
        use_pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments
        
    Returns:
        Configured MedicalHybridModel
    """
    # Default configurations for different model sizes
    size_configs = {
        'small': {
            'n_qubits': 4,
            'n_quantum_layers': 1,
            'classical_backbone': 'custom',
        },
        'medium': {
            'n_qubits': 8,
            'n_quantum_layers': 2,
            'classical_backbone': 'mobilenet',
        },
        'large': {
            'n_qubits': 12,
            'n_quantum_layers': 3,
            'classical_backbone': 'efficientnet',
        }
    }
    
    config = size_configs.get(model_size, size_configs['medium'])
    config.update(kwargs)
    
    # Create and return the model
    model = MedicalHybridModel(
        input_shape=input_shape,
        n_classes=n_classes,
        model_size=model_size,
        use_pretrained=use_pretrained,
        **config
    )
    
    logger.info(f"Created hybrid medical model: {model_size} size, "
               f"{config['classical_backbone']} backbone")
    
    return model
