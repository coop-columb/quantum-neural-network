"""
Integration test for medical imaging application.
"""

import numpy as np
import pytest
import tensorflow as tf

from quantum_nn.applications.medical_imaging import (
    ImageProcessor,
    DataPipeline,
    create_medical_quantum_classifier,
    create_hybrid_medical_model,
)


class TestMedicalImagingIntegration:
    """Integration tests for medical imaging workflow."""

    def test_end_to_end_quantum_workflow(self):
        """Test complete workflow with quantum classifier."""
        # Create synthetic data
        n_samples = 20
        images = np.random.random((n_samples, 64, 64, 3))
        labels = np.random.randint(0, 2, (n_samples,))

        # Initialize processor
        processor = ImageProcessor(
            target_size=(64, 64),  # Smaller for faster testing
            quantum_target_size=(8, 8),
        )

        # Process data
        processed_images = processor.preprocess_batch(images)
        features = processor.extract_features(processed_images)
        quantum_features = processor.prepare_for_quantum(features)

        # Create quantum classifier
        classifier = create_medical_quantum_classifier(
            input_shape=quantum_features.shape[1:],
            n_classes=2,
            n_qubits=4,  # Small for testing
            circuit_complexity="simple",
        )

        # Compile model
        classifier.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train for a few steps (just to verify it works)
        history = classifier.fit(
            quantum_features, labels, epochs=2, batch_size=4, verbose=0
        )

        # Check that training completed
        assert "loss" in history.history
        assert len(history.history["loss"]) == 2

        # Test prediction
        predictions = classifier.predict(quantum_features[:5])
        assert predictions.shape == (5, 1)

    def test_end_to_end_hybrid_workflow(self):
        """Test complete workflow with hybrid model."""
        # Create synthetic data
        n_samples = 16
        images = np.random.random((n_samples, 32, 32, 3))
        labels = np.random.randint(0, 2, (n_samples,))

        # Create data pipeline
        processor = ImageProcessor(target_size=(32, 32))
        pipeline = DataPipeline(
            image_processor=processor, batch_size=4, validation_split=0.25
        )

        # Create datasets
        train_ds, val_ds, class_weights = pipeline.create_datasets(images, labels)

        # Create hybrid model
        model = create_hybrid_medical_model(
            input_shape=(32, 32, 3),
            n_classes=2,
            model_size="small",
            use_pretrained=False,  # Avoid downloading weights in tests
        )

        # Compile model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train for a few steps
        history = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)

        # Check that training completed
        assert "loss" in history.history
        assert len(history.history["loss"]) == 1

        # Test prediction
        test_images = np.random.random((2, 32, 32, 3))
        predictions = model.predict(test_images, verbose=0)
        assert predictions.shape == (2, 1)

    def test_preprocessing_pipeline_consistency(self):
        """Test that preprocessing pipeline produces consistent results."""
        # Create test images
        images = np.random.random((10, 50, 50, 3))

        # Process twice
        processor = ImageProcessor()
        result1 = processor.preprocess_batch(images)
        result2 = processor.preprocess_batch(images)

        # Results should be identical (deterministic preprocessing)
        np.testing.assert_array_equal(result1, result2)

        # Feature extraction should also be consistent
        features1 = processor.extract_features(result1)
        features2 = processor.extract_features(result2)

        np.testing.assert_array_almost_equal(features1, features2, decimal=5)
