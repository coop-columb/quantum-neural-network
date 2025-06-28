"""
Data pipeline for medical imaging quantum neural network.

This module handles dataset creation, batching, and feeding data
to the quantum neural network model.
"""
import os
import glob
import random
from typing import Tuple, Dict, List, Optional, Union
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

from ..data.config import DataConfig
from .image_processor import MedicalImageProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImageDataPipeline:
    """Data pipeline for chest X-ray images."""

    def __init__(
            self,
            config: Optional[DataConfig] = None,
            image_processor: Optional[MedicalImageProcessor] = None
    ):
        """
        Initialize the data pipeline.

        Args:
            config: Data configuration object
            image_processor: Image processor instance
        """
        self.config = config or DataConfig()
        self.processor = image_processor or MedicalImageProcessor(self.config)
        self.dataset_path = Path(self.config.raw_data_dir) / "chest_xray"

        # Cache for dataset statistics
        self._dataset_info = None

    def get_image_paths_and_labels(
            self,
            split: str = "train"
    ) -> Tuple[List[str], List[int]]:
        """
        Get all image paths and labels for a given split.

        Args:
            split: Dataset split ('train', 'val', or 'test')

        Returns:
            Tuple of (image_paths, labels)
        """
        split_path = self.dataset_path / split

        if not split_path.exists():
            raise FileNotFoundError(f"Split directory not found: {split_path}")

        image_paths = []
        labels = []

        # Get normal images (label 0)
        normal_path = split_path / "NORMAL"
        normal_images = glob.glob(str(normal_path / "*.jpeg"))
        normal_images.extend(glob.glob(str(normal_path / "*.jpg")))
        image_paths.extend(normal_images)
        labels.extend([0] * len(normal_images))

        # Get pneumonia images (label 1)
        pneumonia_path = split_path / "PNEUMONIA"
        pneumonia_images = glob.glob(str(pneumonia_path / "*.jpeg"))
        pneumonia_images.extend(glob.glob(str(pneumonia_path / "*.jpg")))
        image_paths.extend(pneumonia_images)
        labels.extend([1] * len(pneumonia_images))

        logger.info(f"{split} set: {len(normal_images)} normal, {len(pneumonia_images)} pneumonia")

        return image_paths, labels

    def create_balanced_validation_set(
            self,
            train_paths: List[str],
            train_labels: List[int],
            val_size: float = 0.15
    ) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Create a balanced validation set from training data.

        Since the original validation set only has 16 images, we create
        a proper validation set from the training data.

        Args:
            train_paths: Original training image paths
            train_labels: Original training labels
            val_size: Proportion of training data to use for validation

        Returns:
            Tuple of (new_train_paths, new_train_labels, val_paths, val_labels)
        """
        # Split training data stratified by class
        new_train_paths, val_paths, new_train_labels, val_labels = train_test_split(
            train_paths,
            train_labels,
            test_size=val_size,
            stratify=train_labels,
            random_state=self.config.random_seed
        )

        logger.info(f"Created validation set: {len(val_paths)} images from training data")
        logger.info(f"New training set size: {len(new_train_paths)} images")

        return new_train_paths, new_train_labels, val_paths, val_labels

    def create_tf_dataset(
            self,
            image_paths: List[str],
            labels: List[int],
            batch_size: Optional[int] = None,
            shuffle: bool = True,
            augment: bool = False,
            cache: bool = True
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from image paths and labels.

        Args:
            image_paths: List of image file paths
            labels: List of labels
            batch_size: Batch size (uses config default if None)
            shuffle: Whether to shuffle the dataset
            augment: Whether to apply data augmentation
            cache: Whether to cache the dataset

        Returns:
            TensorFlow dataset
        """
        batch_size = batch_size or self.config.batch_size

        # Create dataset from paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))

        # Process in batches for efficiency
        def process_batch(paths_batch, labels_batch):
            """Process a batch of images."""
            # Convert tensor to numpy for processing
            paths = [p.decode('utf-8') for p in paths_batch.numpy()]
            labels_np = labels_batch.numpy()

            # Process with our image processor
            quantum_images, classical_features, original_images, labels_out = \
                self.processor.preprocess_batch(paths, labels_np, augment=augment)

            # Flatten quantum images for encoding
            quantum_flat = quantum_images.reshape(quantum_images.shape[0], -1)

            return (
                quantum_flat.astype(np.float32),
                classical_features.astype(np.float32),
                labels_out.astype(np.int32)
            )

        # Map processing function
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(
            lambda paths, labels: tf.py_function(
                process_batch,
                [paths, labels],
                [tf.float32, tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Cache if requested
        if cache:
            cache_dir = Path(self.config.cache_dir) / "tf_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            dataset = dataset.cache(str(cache_dir))

        # Prefetch for performance
        dataset = dataset.prefetch(self.config.prefetch_buffer)

        return dataset

    def prepare_datasets(
            self,
            use_original_val: bool = False,
            val_size: float = 0.15,
            limit_samples: Optional[int] = None
    ) -> Dict[str, tf.data.Dataset]:
        """
        Prepare all datasets for training, validation, and testing.

        Args:
            use_original_val: Whether to use the small original validation set
            val_size: Size of validation set if creating from training data
            limit_samples: Limit number of samples for development

        Returns:
            Dictionary with 'train', 'val', and 'test' datasets
        """
        datasets = {}

        # Get training data
        train_paths, train_labels = self.get_image_paths_and_labels("train")

        # Handle validation set
        if use_original_val:
            val_paths, val_labels = self.get_image_paths_and_labels("val")
        else:
            # Create balanced validation set from training data
            train_paths, train_labels, val_paths, val_labels = \
                self.create_balanced_validation_set(train_paths, train_labels, val_size)

        # Get test data
        test_paths, test_labels = self.get_image_paths_and_labels("test")

        # Limit samples if requested (for development)
        if limit_samples:
            logger.info(f"Limiting dataset to {limit_samples} samples per split")
            train_paths = train_paths[:limit_samples]
            train_labels = train_labels[:limit_samples]
            val_paths = val_paths[:min(limit_samples, len(val_paths))]
            val_labels = val_labels[:min(limit_samples, len(val_labels))]
            test_paths = test_paths[:min(limit_samples, len(test_paths))]
            test_labels = test_labels[:min(limit_samples, len(test_labels))]

        # Create TensorFlow datasets
        datasets['train'] = self.create_tf_dataset(
            train_paths, train_labels,
            shuffle=True, augment=True
        )

        datasets['val'] = self.create_tf_dataset(
            val_paths, val_labels,
            shuffle=False, augment=False
        )

        datasets['test'] = self.create_tf_dataset(
            test_paths, test_labels,
            shuffle=False, augment=False
        )

        # Store dataset info
        self._dataset_info = {
            'train_size': len(train_paths),
            'val_size': len(val_paths),
            'test_size': len(test_paths),
            'train_balance': sum(train_labels) / len(train_labels),
            'val_balance': sum(val_labels) / len(val_labels),
            'test_balance': sum(test_labels) / len(test_labels)
        }

        self._log_dataset_info()

        return datasets

    def _log_dataset_info(self):
        """Log dataset information."""
        if not self._dataset_info:
            return

        logger.info("Dataset Summary:")
        logger.info(f"  Training: {self._dataset_info['train_size']} images "
                    f"({self._dataset_info['train_balance']:.1%} pneumonia)")
        logger.info(f"  Validation: {self._dataset_info['val_size']} images "
                    f"({self._dataset_info['val_balance']:.1%} pneumonia)")
        logger.info(f"  Test: {self._dataset_info['test_size']} images "
                    f"({self._dataset_info['test_balance']:.1%} pneumonia)")

    def get_class_weights(self) -> Dict[int, float]:
        """
        Calculate class weights to handle imbalance.

        Returns:
            Dictionary mapping class indices to weights
        """
        if not self._dataset_info:
            raise ValueError("Must call prepare_datasets first")

        # Calculate based on training set balance
        pneumonia_ratio = self._dataset_info['train_balance']
        normal_ratio = 1 - pneumonia_ratio

        # Inverse frequency weighting
        weights = {
            0: 1.0 / normal_ratio,  # Normal class weight
            1: 1.0 / pneumonia_ratio  # Pneumonia class weight
        }

        # Normalize so average weight is 1
        avg_weight = sum(weights.values()) / len(weights)
        weights = {k: v / avg_weight for k, v in weights.items()}

        logger.info(f"Class weights: Normal={weights[0]:.3f}, Pneumonia={weights[1]:.3f}")

        return weights

    def create_sample_batch(self, n_samples: int = 5) -> Dict[str, np.ndarray]:
        """
        Create a sample batch for testing the pipeline.

        Args:
            n_samples: Number of samples to include

        Returns:
            Dictionary with sample data
        """
        # Get a few training images
        train_paths, train_labels = self.get_image_paths_and_labels("train")

        # Sample randomly
        indices = random.sample(range(len(train_paths)), min(n_samples, len(train_paths)))
        sample_paths = [train_paths[i] for i in indices]
        sample_labels = [train_labels[i] for i in indices]

        # Process the batch
        quantum_images, classical_features, original_images, labels = \
            self.processor.preprocess_batch(sample_paths, sample_labels, augment=False)

        return {
            'quantum_images': quantum_images,
            'classical_features': classical_features,
            'original_images': original_images,
            'labels': labels,
            'paths': sample_paths
        }


def test_pipeline():
    """Test the data pipeline with a small batch."""
    logger.info("Testing data pipeline...")

    # Create pipeline
    pipeline = MedicalImageDataPipeline()

    # Create sample batch
    sample = pipeline.create_sample_batch(n_samples=3)

    logger.info(f"Sample batch created:")
    logger.info(f"  Quantum images shape: {sample['quantum_images'].shape}")
    logger.info(f"  Classical features shape: {sample['classical_features'].shape}")
    logger.info(f"  Labels: {sample['labels']}")

    # Test dataset creation
    logger.info("\nTesting dataset creation...")
    datasets = pipeline.prepare_datasets(limit_samples=10)

    # Test iteration
    for batch in datasets['train'].take(1):
        quantum_flat, classical_features, labels = batch
        logger.info(f"  Batch shapes: quantum={quantum_flat.shape}, "
                    f"features={classical_features.shape}, labels={labels.shape}")

    logger.info("Pipeline test complete!")


if __name__ == "__main__":
    test_pipeline()