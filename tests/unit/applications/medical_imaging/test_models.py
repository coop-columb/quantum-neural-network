"""
Tests for medical imaging models.
"""
import numpy as np
import pytest
import tensorflow as tf

from quantum_nn.applications.medical_imaging.models import (
    MedicalQuantumClassifier,
    MedicalHybridModel,
    create_medical_quantum_classifier,
    create_hybrid_medical_model
)


class TestMedicalQuantumClassifier:
    """Test suite for medical quantum classifier."""

    def test_initialization(self):
        """Test quantum classifier initialization."""
        classifier = MedicalQuantumClassifier(
            n_qubits=4,
            n_layers=2,
            circuit_type="strongly_entangling",
            encoding_type="angle",
            n_classes=2
        )
        
        assert classifier.n_qubits == 4
        assert classifier.n_layers == 2
        assert classifier.circuit_type == "strongly_entangling"
        assert classifier.encoding_type == "angle"
        assert classifier.n_classes == 2
    
    def test_factory_function(self):
        """Test the factory function for creating quantum classifiers."""
        classifier = create_medical_quantum_classifier(
            input_shape=(10,),
            n_classes=2,
            circuit_complexity="simple"
        )
        
        assert isinstance(classifier, MedicalQuantumClassifier)
        assert classifier.n_classes == 2
    
    def test_forward_pass(self):
        """Test forward pass through the quantum classifier."""
        classifier = MedicalQuantumClassifier(
            n_qubits=4,
            n_layers=1,
            n_classes=2
        )
        
        # Create dummy input
        batch_size = 2
        input_dim = 8
        inputs = tf.random.normal((batch_size, input_dim))
        
        # Forward pass
        outputs = classifier(inputs)
        
        # Check output shape
        assert outputs.shape == (batch_size, 1)  # Binary classification
        
        # Check output range for sigmoid activation
        assert tf.reduce_all(outputs >= 0)
        assert tf.reduce_all(outputs <= 1)
    
    def test_config_serialization(self):
        """Test model configuration serialization."""
        classifier = MedicalQuantumClassifier(
            n_qubits=6,
            n_layers=3,
            circuit_type="convolution",
            encoding_type="hybrid"
        )
        
        config = classifier.get_config()
        
        assert config['n_qubits'] == 6
        assert config['n_layers'] == 3
        assert config['circuit_type'] == "convolution"
        assert config['encoding_type'] == "hybrid"


class TestMedicalHybridModel:
    """Test suite for medical hybrid model."""

    def test_initialization(self):
        """Test hybrid model initialization."""
        model = MedicalHybridModel(
            input_shape=(64, 64, 3),
            n_classes=3,
            classical_backbone="mobilenet",
            n_qubits=6,
            n_quantum_layers=2
        )
        
        assert model.input_shape == (64, 64, 3)
        assert model.n_classes == 3
        assert model.classical_backbone == "mobilenet"
        assert model.n_qubits == 6
        assert model.n_quantum_layers == 2
    
    def test_factory_function(self):
        """Test the factory function for creating hybrid models."""
        model = create_hybrid_medical_model(
            input_shape=(32, 32, 3),
            n_classes=2,
            model_size="small"
        )
        
        assert isinstance(model, MedicalHybridModel)
        assert model.n_classes == 2
    
    def test_forward_pass(self):
        """Test forward pass through the hybrid model."""
        model = MedicalHybridModel(
            input_shape=(32, 32, 3),
            n_classes=2,
            n_qubits=4,
            n_quantum_layers=1,
            use_pretrained=False  # Avoid downloading weights in tests
        )
        
        # Create dummy input
        batch_size = 2
        inputs = tf.random.normal((batch_size, 32, 32, 3))
        
        # Forward pass
        outputs = model(inputs)
        
        # Check output shape
        assert outputs.shape == (batch_size, 1)  # Binary classification
        
        # Check output range for sigmoid activation
        assert tf.reduce_all(outputs >= 0)
        assert tf.reduce_all(outputs <= 1)
    
    def test_backbone_freezing(self):
        """Test backbone freezing functionality."""
        model = MedicalHybridModel(
            input_shape=(32, 32, 3),
            n_classes=2,
            freeze_backbone=True,
            use_pretrained=False
        )
        
        # Check that backbone is frozen
        assert not model.backbone.trainable
        
        # Unfreeze and check
        model.unfreeze_backbone()
        assert model.backbone.trainable
    
    def test_fusion_strategies(self):
        """Test different fusion strategies."""
        strategies = ["concatenate", "attention", "gated"]
        
        for strategy in strategies:
            model = MedicalHybridModel(
                input_shape=(32, 32, 3),
                n_classes=2,
                fusion_strategy=strategy,
                use_pretrained=False
            )
            
            assert model.fusion_strategy == strategy
            
            # Test forward pass
            inputs = tf.random.normal((1, 32, 32, 3))
            outputs = model(inputs)
            assert outputs.shape == (1, 1)
    
    def test_config_serialization(self):
        """Test model configuration serialization."""
        model = MedicalHybridModel(
            input_shape=(64, 64, 3),
            n_classes=3,
            classical_backbone="resnet",
            fusion_strategy="attention"
        )
        
        config = model.get_config()
        
        assert config['input_shape'] == (64, 64, 3)
        assert config['n_classes'] == 3
        assert config['classical_backbone'] == "resnet"
        assert config['fusion_strategy'] == "attention"
