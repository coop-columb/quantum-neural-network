#!/usr/bin/env python
"""
Medical imaging demonstration script.

This script demonstrates how to use the quantum neural network framework
for medical image classification tasks.
"""
import os
import sys
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from quantum_nn.applications.medical_imaging import (
    ImageProcessor,
    DataPipeline,
    MedicalQuantumClassifier,
    MedicalHybridModel,
    create_medical_quantum_classifier,
    create_hybrid_medical_model,
)
from quantum_nn.benchmarks import run_classical_comparison

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_medical_data(
    n_samples: int = 1000,
    image_size: Tuple[int, int] = (224, 224),
    n_classes: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic medical imaging data for demonstration.

    Args:
        n_samples: Number of samples to generate
        image_size: Size of images (H, W)
        n_classes: Number of classes
        seed: Random seed

    Returns:
        Tuple of (images, labels)
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Generate synthetic images with different patterns for each class
    images = []
    labels = []

    for i in range(n_samples):
        # Create base image with noise
        img = np.random.normal(0.5, 0.1, (*image_size, 3))

        # Add class-specific patterns
        class_label = i % n_classes

        if class_label == 0:
            # Class 0: Add circular patterns (simulating normal tissue)
            center_x, center_y = image_size[0] // 2, image_size[1] // 2
            y, x = np.ogrid[: image_size[0], : image_size[1]]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (
                image_size[0] // 4
            ) ** 2
            img[mask] += 0.3

        else:
            # Class 1: Add irregular patterns (simulating abnormal tissue)
            for _ in range(5):
                x = np.random.randint(0, image_size[0])
                y = np.random.randint(0, image_size[1])
                size = np.random.randint(10, 30)
                img[
                    max(0, x - size) : min(image_size[0], x + size),
                    max(0, y - size) : min(image_size[1], y + size),
                ] += 0.4

        # Clip values to valid range
        img = np.clip(img, 0, 1)

        images.append(img)
        labels.append(class_label)

    return np.array(images), np.array(labels)


def demonstrate_preprocessing():
    """Demonstrate the preprocessing pipeline."""
    logger.info("=== Preprocessing Demonstration ===")

    # Create synthetic data
    images, labels = create_synthetic_medical_data(n_samples=100)
    logger.info(f"Created {len(images)} synthetic medical images")

    # Initialize image processor
    processor = ImageProcessor(target_size=(224, 224), quantum_target_size=(28, 28))

    # Process images
    processed_images = processor.preprocess_batch(images)
    features = processor.extract_features(processed_images)
    quantum_features = processor.prepare_for_quantum(features)

    logger.info(f"Original images shape: {images.shape}")
    logger.info(f"Processed images shape: {processed_images.shape}")
    logger.info(f"Extracted features shape: {features.shape}")
    logger.info(f"Quantum features shape: {quantum_features.shape}")

    # Create data pipeline
    pipeline = DataPipeline(
        image_processor=processor, batch_size=16, validation_split=0.2
    )

    # Create datasets
    train_ds, val_ds, class_weights = pipeline.create_datasets(images, labels)

    logger.info(f"Training dataset created with class weights: {class_weights}")
    logger.info("Preprocessing demonstration completed\n")

    return quantum_features, labels, train_ds, val_ds, class_weights


def demonstrate_quantum_classifier(x_train, y_train, x_val, y_val):
    """Demonstrate the quantum classifier."""
    logger.info("=== Quantum Classifier Demonstration ===")

    # Create quantum classifier
    quantum_model = create_medical_quantum_classifier(
        input_shape=x_train.shape[1:], n_classes=2, circuit_complexity="medium", seed=42
    )

    # Compile for medical imaging
    quantum_model.compile_for_medical_imaging()

    logger.info("Quantum classifier architecture:")
    quantum_model.summary()

    # Train the model (small number of epochs for demo)
    logger.info("Training quantum classifier...")
    history = quantum_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=5,  # Small number for demo
        batch_size=16,
        verbose=1,
    )

    # Evaluate performance
    logger.info("Evaluating quantum classifier...")
    metrics = quantum_model.evaluate_medical_performance(
        x_val, y_val, class_names=["Normal", "Abnormal"]
    )

    logger.info("Quantum classifier performance:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")

    logger.info("Quantum classifier demonstration completed\n")
    return quantum_model, history


def demonstrate_hybrid_model(train_ds, val_ds):
    """Demonstrate the hybrid classical-quantum model."""
    logger.info("=== Hybrid Model Demonstration ===")

    # Create hybrid model
    hybrid_model = create_hybrid_medical_model(
        input_shape=(224, 224, 3),
        n_classes=2,
        model_size="small",  # Use small for demo
        fusion_strategy="concatenate",
        freeze_backbone=True,  # Start with frozen backbone
        seed=42,
    )

    # Compile for medical imaging
    hybrid_model.compile_for_medical_imaging()

    logger.info("Hybrid model architecture:")
    hybrid_model.summary()

    # Progressive training (small epochs for demo)
    logger.info("Training hybrid model with progressive strategy...")
    histories = hybrid_model.fit_with_progressive_training(
        train_ds,
        validation_data=val_ds,
        initial_epochs=2,  # Small numbers for demo
        fine_tune_epochs=3,
        batch_size=16,
    )

    logger.info("Hybrid model demonstration completed\n")
    return hybrid_model, histories


def demonstrate_comparison(quantum_model, hybrid_model, x_train, y_train, x_val, y_val):
    """Demonstrate comparison between models."""
    logger.info("=== Model Comparison Demonstration ===")

    # Run comparison (simplified for demo)
    logger.info("Comparing quantum model performance...")

    # Evaluate both models
    quantum_metrics = quantum_model.evaluate_medical_performance(
        x_val, y_val, class_names=["Normal", "Abnormal"]
    )

    # For hybrid model, we need to prepare the data differently
    # (This is simplified - in practice you'd use the full pipeline)

    logger.info("Model comparison:")
    logger.info("Quantum Model Performance:")
    for metric, value in quantum_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")

    logger.info("Model comparison demonstration completed\n")


def demonstrate_interpretability(quantum_model, x_sample):
    """Demonstrate interpretability features."""
    logger.info("=== Interpretability Demonstration ===")

    # Get feature importance
    importance_scores = quantum_model.get_quantum_feature_importance(
        x_sample[:1], method="gradient"  # Use first sample
    )

    logger.info(f"Feature importance scores shape: {importance_scores.shape}")
    logger.info(f"Top 5 most important features: {np.argsort(importance_scores)[-5:]}")

    logger.info("Interpretability demonstration completed\n")


def main():
    """Main demonstration function."""
    logger.info("Starting Medical Imaging Quantum Neural Network Demonstration")
    logger.info("=" * 60)

    try:
        # Set seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Demonstrate preprocessing
        quantum_features, labels, train_ds, val_ds, class_weights = (
            demonstrate_preprocessing()
        )

        # Convert labels to one-hot for training
        y_train = tf.keras.utils.to_categorical(labels, 2)

        # Split data for quantum model (simplified)
        split_idx = int(0.8 * len(quantum_features))
        x_train_q = quantum_features[:split_idx]
        y_train_q = y_train[:split_idx]
        x_val_q = quantum_features[split_idx:]
        y_val_q = y_train[split_idx:]

        # Demonstrate quantum classifier
        quantum_model, quantum_history = demonstrate_quantum_classifier(
            x_train_q, y_train_q, x_val_q, y_val_q
        )

        # Demonstrate hybrid model
        hybrid_model, hybrid_histories = demonstrate_hybrid_model(train_ds, val_ds)

        # Demonstrate comparison
        demonstrate_comparison(
            quantum_model, hybrid_model, x_train_q, y_train_q, x_val_q, y_val_q
        )

        # Demonstrate interpretability
        demonstrate_interpretability(quantum_model, x_val_q)

        logger.info("=" * 60)
        logger.info("Medical Imaging Demonstration Completed Successfully!")
        logger.info("\nKey Features Demonstrated:")
        logger.info("✓ Medical image preprocessing pipeline")
        logger.info("✓ Quantum neural network classifier")
        logger.info("✓ Hybrid classical-quantum model")
        logger.info("✓ Progressive training strategy")
        logger.info("✓ Medical-specific evaluation metrics")
        logger.info("✓ Interpretability features")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
