"""
Hybrid classical-quantum model for medical imaging applications.

This module implements a sophisticated hybrid architecture that combines
the power of classical deep learning with quantum computing advantages
for medical image analysis.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging

import numpy as np
import tensorflow as tf
import pennylane as qml

from quantum_nn.circuits import ParameterizedCircuit
from quantum_nn.circuits.templates import StronglyEntanglingLayers
from quantum_nn.circuits.encodings import AngleEncoding, AmplitudeEncoding
from quantum_nn.layers import QuantumLayer
from quantum_nn.models import QuantumModel
from quantum_nn.optimizers import ParameterShiftOptimizer, QuantumNaturalGradient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalHybridModel(tf.keras.Model):
    """
    Hybrid classical-quantum model for medical imaging.
    
    This model combines classical convolutional neural networks for initial
    feature extraction with quantum neural networks for high-level pattern
    recognition and classification. The architecture is specifically designed
    for medical imaging applications where both spatial features and complex
    correlations are important.
    
    Architecture:
    1. Classical CNN backbone for spatial feature extraction
    2. Quantum feature processing for complex pattern recognition
    3. Hybrid fusion layer for combining classical and quantum features
    4. Medical-specific output layers with appropriate activations
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_classes: int = 2,
        classical_backbone: str = "mobilenet",
        n_qubits: int = 8,
        n_quantum_layers: int = 3,
        fusion_strategy: str = "concatenate",
        dropout_rate: float = 0.3,
        quantum_device: str = "default.qubit",
        use_pretrained: bool = True,
        freeze_backbone: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the hybrid medical model.
        
        Args:
            input_shape: Shape of input images (H, W, C)
            n_classes: Number of output classes
            classical_backbone: Classical CNN backbone ('mobilenet', 'resnet', 'efficientnet')
            n_qubits: Number of qubits in quantum circuit
            n_quantum_layers: Number of quantum layers
            fusion_strategy: Strategy for fusing classical and quantum features
            dropout_rate: Dropout rate for regularization
            quantum_device: PennyLane quantum device
            use_pretrained: Whether to use pretrained classical backbone
            freeze_backbone: Whether to freeze backbone weights initially
            seed: Random seed for reproducibility
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.classical_backbone = classical_backbone
        self.n_qubits = n_qubits
        self.n_quantum_layers = n_quantum_layers
        self.fusion_strategy = fusion_strategy
        self.dropout_rate = dropout_rate
        self.quantum_device = quantum_device
        self.use_pretrained = use_pretrained
        self.freeze_backbone = freeze_backbone
        
        # Build the hybrid architecture
        self._build_classical_backbone()
        self._build_quantum_processor()
        self._build_fusion_layers()
        self._build_output_layers()
        
        # Initialize medical metrics
        self._initialize_medical_metrics()
        
        logger.info(f"Initialized MedicalHybridModel with {classical_backbone} backbone "
                   f"and {n_qubits}-qubit quantum processor")
    
    def _build_classical_backbone(self):
        """Build the classical CNN backbone for feature extraction."""
        if self.classical_backbone == "mobilenet":
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=self.input_shape,
                weights='imagenet' if self.use_pretrained else None,
                include_top=False,
                pooling='avg'
            )
        elif self.classical_backbone == "resnet":
            base_model = tf.keras.applications.ResNet50V2(
                input_shape=self.input_shape,
                weights='imagenet' if self.use_pretrained else None,
                include_top=False,
                pooling='avg'
            )
        elif self.classical_backbone == "efficientnet":
            base_model = tf.keras.applications.EfficientNetB0(
                input_shape=self.input_shape,
                weights='imagenet' if self.use_pretrained else None,
                include_top=False,
                pooling='avg'
            )
        else:
            raise ValueError(f"Unknown backbone: {self.classical_backbone}")
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            base_model.trainable = False
            logger.info("Frozen classical backbone for initial training")
        
        self.backbone = base_model
        
        # Add classical feature processing layers
        self.classical_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(
                256, 
                activation='relu', 
                name='classical_dense_1',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.Dropout(self.dropout_rate, name='classical_dropout_1'),
            tf.keras.layers.Dense(
                128, 
                activation='relu', 
                name='classical_dense_2',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.Dropout(self.dropout_rate / 2, name='classical_dropout_2'),
        ], name='classical_processor')
        
        # Get classical feature dimension
        self.classical_feature_dim = 128
    
    def _build_quantum_processor(self):
        """Build the quantum neural network processor."""
        # Create quantum circuit for feature processing
        quantum_template = StronglyEntanglingLayers(
            self.n_qubits, 
            self.n_quantum_layers,
            pattern="circular"
        )
        
        # Use angle encoding for classical features
        quantum_encoder = AngleEncoding(
            self.n_qubits, 
            rotation="Y",
            scaling=np.pi / 4  # Scale features to appropriate range
        )
        
        self.quantum_circuit = ParameterizedCircuit(
            n_qubits=self.n_qubits,
            template=quantum_template,
            encoder=quantum_encoder,
            device=self.quantum_device
        )
        
        # Create quantum layer
        self.quantum_layer = QuantumLayer(
            circuit=self.quantum_circuit,
            weight_shape=(self.quantum_circuit.get_n_params(),),
            measurement_type="expectation"
        )
        
        # Add quantum feature processing
        self.quantum_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_qubits, 
                activation='tanh',  # Bounded activation for quantum encoding
                name='quantum_encoder',
                kernel_initializer='glorot_uniform'
            ),
            self.quantum_layer,
            tf.keras.layers.Dense(
                64, 
                activation='relu', 
                name='quantum_dense_1',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.Dropout(self.dropout_rate / 2, name='quantum_dropout_1'),
        ], name='quantum_processor')
        
        # Get quantum feature dimension
        self.quantum_feature_dim = 64
    
    def _build_fusion_layers(self):
        """Build layers for fusing classical and quantum features."""
        if self.fusion_strategy == "concatenate":
            # Simple concatenation
            self.fusion_dim = self.classical_feature_dim + self.quantum_feature_dim
            self.fusion_layer = tf.keras.layers.Concatenate(name='feature_fusion')
            
        elif self.fusion_strategy == "attention":
            # Attention-based fusion
            self.fusion_dim = max(self.classical_feature_dim, self.quantum_feature_dim)
            
            # Attention mechanism
            self.classical_attention = tf.keras.layers.Dense(
                self.fusion_dim, 
                activation='tanh',
                name='classical_attention'
            )
            self.quantum_attention = tf.keras.layers.Dense(
                self.fusion_dim, 
                activation='tanh',
                name='quantum_attention'
            )
            self.attention_weights = tf.keras.layers.Dense(
                2, 
                activation='softmax',
                name='attention_weights'
            )
            
        elif self.fusion_strategy == "gated":
            # Gated fusion mechanism
            self.fusion_dim = self.classical_feature_dim + self.quantum_feature_dim
            
            self.gate_layer = tf.keras.layers.Dense(
                self.fusion_dim, 
                activation='sigmoid',
                name='fusion_gate'
            )
            
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        # Post-fusion processing
        self.fusion_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(
                128, 
                activation='relu', 
                name='fusion_dense_1',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.Dropout(self.dropout_rate, name='fusion_dropout_1'),
            tf.keras.layers.Dense(
                64, 
                activation='relu', 
                name='fusion_dense_2',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.Dropout(self.dropout_rate / 2, name='fusion_dropout_2'),
        ], name='fusion_processor')
    
    def _build_output_layers(self):
        """Build the output layers for medical classification."""
        if self.n_classes == 2:
            # Binary classification
            self.output_layer = tf.keras.layers.Dense(
                1, 
                activation='sigmoid', 
                name='medical_output'
            )
        else:
            # Multi-class classification
            self.output_layer = tf.keras.layers.Dense(
                self.n_classes, 
                activation='softmax', 
                name='medical_output'
            )
    
    def _initialize_medical_metrics(self):
        """Initialize medical-specific metrics."""
        if self.n_classes == 2:
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
            self.metrics = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.F1Score(name='f1_score'),
            ]
        else:
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
            self.metrics = [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                tf.keras.metrics.F1Score(average='weighted', name='f1_score'),
            ]
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            inputs: Input tensor of shape (batch_size, H, W, C)
            training: Whether in training mode
            
        Returns:
            Output predictions
        """
        # Classical feature extraction
        classical_features = self.backbone(inputs, training=training)
        classical_features = self.classical_processor(classical_features, training=training)
        
        # Quantum feature processing
        # Use classical features as input to quantum processor
        quantum_features = self.quantum_processor(classical_features, training=training)
        
        # Feature fusion
        if self.fusion_strategy == "concatenate":
            fused_features = self.fusion_layer([classical_features, quantum_features])
            
        elif self.fusion_strategy == "attention":
            # Apply attention mechanism
            classical_att = self.classical_attention(classical_features)
            quantum_att = self.quantum_attention(quantum_features)
            
            # Compute attention weights
            combined = tf.concat([classical_att, quantum_att], axis=-1)
            weights = self.attention_weights(combined)
            
            # Apply attention
            classical_weighted = classical_att * weights[:, 0:1]
            quantum_weighted = quantum_att * weights[:, 1:2]
            fused_features = classical_weighted + quantum_weighted
            
        elif self.fusion_strategy == "gated":
            # Apply gated fusion
            concatenated = tf.concat([classical_features, quantum_features], axis=-1)
            gate = self.gate_layer(concatenated)
            fused_features = concatenated * gate
        
        # Post-fusion processing
        processed_features = self.fusion_processor(fused_features, training=training)
        
        # Output prediction
        outputs = self.output_layer(processed_features)
        
        return outputs
    
    def unfreeze_backbone(self):
        """Unfreeze the classical backbone for fine-tuning."""
        self.backbone.trainable = True
        logger.info("Unfrozen classical backbone for fine-tuning")
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'input_shape': self.input_shape,
            'n_classes': self.n_classes,
            'classical_backbone': self.classical_backbone,
            'n_qubits': self.n_qubits,
            'n_quantum_layers': self.n_quantum_layers,
            'fusion_strategy': self.fusion_strategy,
            'dropout_rate': self.dropout_rate,
            'quantum_device': self.quantum_device,
            'use_pretrained': self.use_pretrained,
            'freeze_backbone': self.freeze_backbone,
        }


def create_hybrid_medical_model(
    input_shape: Tuple[int, ...],
    n_classes: int = 2,
    model_size: str = "medium",
    fusion_strategy: str = "concatenate",
    **kwargs
) -> MedicalHybridModel:
    """
    Factory function to create a hybrid medical model.
    
    Args:
        input_shape: Shape of input images
        n_classes: Number of output classes
        model_size: Model size ('small', 'medium', 'large')
        fusion_strategy: Feature fusion strategy
        **kwargs: Additional arguments
        
    Returns:
        Configured MedicalHybridModel
    """
    # Model size configurations
    size_configs = {
        'small': {
            'classical_backbone': 'mobilenet',
            'n_qubits': 6,
            'n_quantum_layers': 2,
            'dropout_rate': 0.2
        },
        'medium': {
            'classical_backbone': 'mobilenet',
            'n_qubits': 8,
            'n_quantum_layers': 3,
            'dropout_rate': 0.3
        },
        'large': {
            'classical_backbone': 'resnet',
            'n_qubits': 10,
            'n_quantum_layers': 4,
            'dropout_rate': 0.4
        }
    }
    
    config = size_configs.get(model_size, size_configs['medium'])
    config.update(kwargs)
    
    # Create and return the model
    model = MedicalHybridModel(
        input_shape=input_shape,
        n_classes=n_classes,
        fusion_strategy=fusion_strategy,
        **config
    )
    
    logger.info(f"Created hybrid medical model: {model_size} size, "
               f"{fusion_strategy} fusion, {config['classical_backbone']} backbone")
    
    return model
