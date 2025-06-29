"""
Tests for medical imaging preprocessing.
"""

import numpy as np
import pytest
import tensorflow as tf

from quantum_nn.applications.medical_imaging.preprocessing import (
    ImageProcessor,
    DataPipeline,
)


class TestImageProcessor:
    """Test suite for image processor."""

    def test_initialization(self):
        """Test image processor initialization."""
        processor = ImageProcessor(target_size=(224, 224), quantum_target_size=(28, 28))

        assert processor.target_size == (224, 224)
        assert processor.quantum_target_size == (28, 28)

    def test_preprocess_single_image(self):
        """Test preprocessing a single image."""
        processor = ImageProcessor()

        # Create dummy image
        image = np.random.random((100, 100, 3))

        # Preprocess
        processed = processor.preprocess_image(image)

        # Check output shape
        assert processed.shape == (224, 224, 3)

        # Check value range
        assert np.all(processed >= 0)
        assert np.all(processed <= 1)

    def test_preprocess_batch(self):
        """Test preprocessing a batch of images."""
        processor = ImageProcessor()

        # Create dummy batch
        batch_size = 5
        images = np.random.random((batch_size, 100, 100, 3))

        # Preprocess
        processed = processor.preprocess_batch(images)

        # Check output shape
        assert processed.shape == (batch_size, 224, 224, 3)

    def test_feature_extraction(self):
        """Test feature extraction using MobileNet."""
        processor = ImageProcessor()

        # Create dummy images (properly sized)
        batch_size = 2
        images = np.random.random((batch_size, 224, 224, 3))

        # Extract features
        features = processor.extract_features(images)

        # Check output shape (MobileNetV2 features)
        assert features.shape == (batch_size, 1280)

        # Check that features are reasonable
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_quantum_preparation(self):
        """Test preparation of features for quantum processing."""
        processor = ImageProcessor(quantum_target_size=(16, 16))

        # Create dummy features
        batch_size = 3
        features = np.random.random((batch_size, 1280))

        # Prepare for quantum
        quantum_features = processor.prepare_for_quantum(features)

        # Check output shape (should be flattened image size)
        assert quantum_features.shape == (batch_size, 16 * 16 * 3)

        # Check value range (should be normalized)
        assert np.all(quantum_features >= -1)
        assert np.all(quantum_features <= 1)


class TestDataPipeline:
    """Test suite for data pipeline."""

    def test_initialization(self):
        """Test data pipeline initialization."""
        processor = ImageProcessor()
        pipeline = DataPipeline(
            image_processor=processor, batch_size=16, validation_split=0.2
        )

        assert pipeline.image_processor == processor
        assert pipeline.batch_size == 16
        assert pipeline.validation_split == 0.2

    def test_create_datasets(self):
        """Test dataset creation."""
        processor = ImageProcessor()
        pipeline = DataPipeline(
            image_processor=processor, batch_size=4, validation_split=0.2
        )

        # Create dummy data
        n_samples = 20
        images = np.random.random((n_samples, 64, 64, 3))
        labels = np.random.randint(0, 2, (n_samples,))

        # Create datasets
        train_ds, val_ds, class_weights = pipeline.create_datasets(images, labels)

        # Check that datasets are created
        assert train_ds is not None
        assert val_ds is not None

        # Check class weights
        assert isinstance(class_weights, dict)
        assert len(class_weights) == 2  # Binary classification

        # Check dataset structure
        for batch in train_ds.take(1):
            images_batch, labels_batch = batch
            assert images_batch.shape[0] <= 4  # Batch size
            assert images_batch.shape[1:] == (224, 224, 3)  # Image shape
            assert labels_batch.shape[0] <= 4  # Batch size

    def test_class_weight_calculation(self):
        """Test class weight calculation for imbalanced datasets."""
        processor = ImageProcessor()
        pipeline = DataPipeline(image_processor=processor)

        # Create imbalanced dataset
        images = np.random.random((100, 32, 32, 3))
        labels = np.concatenate(
            [np.zeros(80), np.ones(20)]  # 80% class 0  # 20% class 1
        )

        # Calculate class weights
        class_weights = pipeline._calculate_class_weights(labels)

        # Class 1 should have higher weight due to imbalance
        assert class_weights[1] > class_weights[0]

        # Weights should be reasonable
        assert class_weights[0] > 0
        assert class_weights[1] > 0
