"""
Quantum classifier for medical imaging applications.

This module implements a quantum neural network specifically designed
for medical image classification, leveraging quantum advantages for
high-dimensional feature spaces typical in medical imaging.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging

import numpy as np
import tensorflow as tf
import pennylane as qml

from quantum_nn.circuits import ParameterizedCircuit
from quantum_nn.circuits.templates import StronglyEntanglingLayers, QuantumConvolutionLayers
from quantum_nn.circuits.encodings import AngleEncoding, AmplitudeEncoding, HybridEncoding
from quantum_nn.layers import QuantumLayer
from quantum_nn.models import QuantumModel
from quantum_nn.optimizers import ParameterShiftOptimizer, QuantumNaturalGradient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalQuantumClassifier(QuantumModel):
    """
    Quantum neural network classifier for medical imaging.
    
    This classifier uses quantum circuits to process medical image features,
    potentially providing quantum advantages for pattern recognition in
    high-dimensional medical data spaces.
    
    Features:
    - Quantum feature encoding for medical image data
    - Configurable quantum circuit architectures
    - Integration with quantum-aware optimizers
    - Medical-specific performance metrics
    - Interpretability features for clinical use
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        circuit_type: str = "strongly_entangling",
        encoding_type: str = "hybrid",
        measurement_type: str = "expectation",
        n_classes: int = 2,
        quantum_device: str = "default.qubit",
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the medical quantum classifier.
        
        Args:
            n_qubits: Number of qubits in the quantum circuit
            n_layers: Number of layers in the quantum circuit
            circuit_type: Type of quantum circuit ('strongly_entangling', 'convolution')
            encoding_type: Data encoding scheme ('angle', 'amplitude', 'hybrid')
            measurement_type: Type of quantum measurement ('expectation', 'probability')
            n_classes: Number of output classes
            quantum_device: PennyLane quantum device
            seed: Random seed for reproducibility
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_type = circuit_type
        self.encoding_type = encoding_type
        self.measurement_type = measurement_type
        self.n_classes = n_classes
        self.quantum_device = quantum_device
        
        # Build the quantum circuit
        self._build_quantum_circuit()
        
        # Build the classical post-processing layers
        self._build_classical_layers()
        
        # Initialize metrics for medical applications
        self._initialize_medical_metrics()
        
        logger.info(f"Initialized MedicalQuantumClassifier with {n_qubits} qubits, "
                   f"{n_layers} layers, {circuit_type} circuit")
    
    def _build_quantum_circuit(self):
        """Build the quantum circuit based on configuration."""
        # Create encoding scheme
        if self.encoding_type == "angle":
            encoder = AngleEncoding(self.n_qubits, rotation="Y")
        elif self.encoding_type == "amplitude":
            encoder = AmplitudeEncoding(self.n_qubits, normalize=True)
        elif self.encoding_type == "hybrid":
            # Use hybrid encoding for better feature representation
            angle_encoder = AngleEncoding(self.n_qubits // 2, rotation="Y")
            amp_encoder = AmplitudeEncoding(self.n_qubits - (self.n_qubits // 2))
            encoder = HybridEncoding(
                self.n_qubits, 
                [angle_encoder, amp_encoder],
                features_per_encoder=[self.n_qubits // 2, self.n_qubits - (self.n_qubits // 2)]
            )
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
        # Create circuit template
        if self.circuit_type == "strongly_entangling":
            template = StronglyEntanglingLayers(
                self.n_qubits, 
                self.n_layers, 
                pattern="circular"  # Good for medical feature correlations
            )
        elif self.circuit_type == "convolution":
            template = QuantumConvolutionLayers(
                self.n_qubits, 
                self.n_layers,
                kernel_size=min(3, self.n_qubits),
                stride=1
            )
        else:
            raise ValueError(f"Unknown circuit type: {self.circuit_type}")
        
        # Create parameterized circuit
        self.quantum_circuit = ParameterizedCircuit(
            n_qubits=self.n_qubits,
            template=template,
            encoder=encoder,
            device=self.quantum_device
        )
        
        # Create quantum layer
        self.quantum_layer = QuantumLayer(
            circuit=self.quantum_circuit,
            weight_shape=(self.quantum_circuit.get_n_params(),),
            measurement_type=self.measurement_type
        )
    
    def _build_classical_layers(self):
        """Build classical post-processing layers."""
        # Determine quantum layer output size
        if self.measurement_type == "expectation":
            quantum_output_size = self.n_qubits
        else:  # probability
            quantum_output_size = 2 ** self.n_qubits
        
        # Classical layers for post-processing quantum outputs
        self.classical_layers = [
            tf.keras.layers.Dense(
                64, 
                activation='relu', 
                name='medical_dense_1',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.Dropout(0.3, name='medical_dropout_1'),
            tf.keras.layers.Dense(
                32, 
                activation='relu', 
                name='medical_dense_2',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.Dropout(0.2, name='medical_dropout_2'),
        ]
        
        # Output layer
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
        # Standard metrics
        if self.n_classes == 2:
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
            self.metrics = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
            ]
        else:
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
            self.metrics = [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
            ]
        
        # Add F1 score (important for medical applications)
        self.metrics.append(tf.keras.metrics.F1Score(name='f1_score'))
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the quantum classifier.
        
        Args:
            inputs: Input tensor of shape (batch_size, feature_dim)
            training: Whether in training mode
            
        Returns:
            Output predictions
        """
        # Ensure inputs are properly shaped for quantum layer
        if len(inputs.shape) > 2:
            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
        
        # Quantum processing
        x = self.quantum_layer(inputs)
        
        # Classical post-processing
        for layer in self.classical_layers:
            x = layer(x, training=training)
        
        # Output
        outputs = self.output_layer(x)
        
        return outputs
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'circuit_type': self.circuit_type,
            'encoding_type': self.encoding_type,
            'measurement_type': self.measurement_type,
            'n_classes': self.n_classes,
            'quantum_device': self.quantum_device,
        }


def create_medical_quantum_classifier(
    input_shape: Tuple[int, ...],
    n_classes: int = 2,
    n_qubits: Optional[int] = None,
    circuit_complexity: str = "medium",
    **kwargs
) -> MedicalQuantumClassifier:
    """
    Factory function to create a medical quantum classifier.
    
    Args:
        input_shape: Shape of input data
        n_classes: Number of output classes
        n_qubits: Number of qubits (auto-calculated if None)
        circuit_complexity: Complexity level ('simple', 'medium', 'complex')
        **kwargs: Additional arguments
        
    Returns:
        Configured MedicalQuantumClassifier
    """
    # Auto-calculate number of qubits based on input dimension
    input_dim = np.prod(input_shape)
    
    if n_qubits is None:
        # Use log scale with minimum of 4 qubits
        n_qubits = max(4, min(12, int(np.log2(input_dim)) + 2))
    
    # Set circuit parameters based on complexity
    complexity_configs = {
        'simple': {'n_layers': 1, 'circuit_type': 'strongly_entangling'},
        'medium': {'n_layers': 3, 'circuit_type': 'strongly_entangling'},
        'complex': {'n_layers': 5, 'circuit_type': 'convolution'},
    }
    
    config = complexity_configs.get(circuit_complexity, complexity_configs['medium'])
    config.update(kwargs)
    
    # Create and return the classifier
    classifier = MedicalQuantumClassifier(
        n_qubits=n_qubits,
        n_classes=n_classes,
        **config
    )
    
    logger.info(f"Created medical quantum classifier: {n_qubits} qubits, "
               f"{config['n_layers']} layers, {circuit_complexity} complexity")
    
    return classifier
