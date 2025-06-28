"""
Data pipeline for medical imaging quantum neural network.

This module handles dataset creation, batching, and feeding data
to the quantum neural network model.
"""
from typing import Tuple, Dict, List, Optional, Union
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import logging

from .image_processor import ImageProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Data pipeline for medical imaging applications.
    
    This class handles the creation of TensorFlow datasets from medical images,
    including proper train/validation splits, class balancing, and batch creation.
    """
    
    def __init__(
        self,
        image_processor: ImageProcessor,
        batch_size: int = 32,
        validation_split: float = 0.2,
        shuffle_buffer: int = 1000,
        prefetch_buffer: int = tf.data.AUTOTUNE
    ):
        """
        Initialize the data pipeline.
        
        Args:
            image_processor: Image processor instance
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            shuffle_buffer: Buffer size for shuffling
            prefetch_buffer: Buffer size for prefetching
        """
        self.image_processor = image_processor
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        
        logger.info(f"DataPipeline initialized with batch_size={batch_size}, "
                   f"validation_split={validation_split}")
    
    def create_datasets(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[int, float]]:
        """
        Create training and validation datasets.
        
        Args:
            images: Training images
            labels: Training labels
            validation_data: Optional separate validation data
            
        Returns:
            Tuple of (train_dataset, val_dataset, class_weights)
        """
        # Calculate class weights for imbalanced datasets
        class_weights = self._calculate_class_weights(labels)
        
        if validation_data is None:
            # Split data into train and validation
            x_train, x_val, y_train, y_val = train_test_split(
                images, labels,
                test_size=self.validation_split,
                stratify=labels,
                random_state=42
            )
        else:
            x_train, y_train = images, labels
            x_val, y_val = validation_data
        
        # Preprocess images
        logger.info("Preprocessing training images...")
        x_train_processed = self.image_processor.preprocess_batch(x_train)
        
        logger.info("Preprocessing validation images...")
        x_val_processed = self.image_processor.preprocess_batch(x_val)
        
        # Create TensorFlow datasets
        train_dataset = self._create_tf_dataset(
            x_train_processed, y_train, is_training=True
        )
        
        val_dataset = self._create_tf_dataset(
            x_val_processed, y_val, is_training=False
        )
        
        logger.info(f"Created datasets - Train: {len(x_train)} samples, "
                   f"Val: {len(x_val)} samples")
        
        return train_dataset, val_dataset, class_weights
    
    def _create_tf_dataset(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        is_training: bool = True
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from images and labels.
        
        Args:
            images: Image array
            labels: Label array
            is_training: Whether this is for training (affects shuffling)
            
        Returns:
            TensorFlow dataset
        """
        # Create dataset from arrays
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        
        # Shuffle if training
        if is_training:
            dataset = dataset.shuffle(self.shuffle_buffer)
        
        # Batch the data
        dataset = dataset.batch(self.batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(self.prefetch_buffer)
        
        return dataset
    
    def _calculate_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Args:
            labels: Array of class labels
            
        Returns:
            Dictionary mapping class indices to weights
        """
        unique_classes = np.unique(labels)
        
        # Use sklearn's compute_class_weight
        class_weights_array = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=labels
        )
        
        # Convert to dictionary
        class_weights = {
            int(cls): float(weight)
            for cls, weight in zip(unique_classes, class_weights_array)
        }
        
        logger.info(f"Calculated class weights: {class_weights}")
        return class_weights
    
    def create_quantum_datasets(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[int, float]]:
        """
        Create datasets specifically for quantum models.
        
        Args:
            images: Training images
            labels: Training labels
            validation_data: Optional separate validation data
            
        Returns:
            Tuple of (train_dataset, val_dataset, class_weights)
        """
        # Calculate class weights
        class_weights = self._calculate_class_weights(labels)
        
        if validation_data is None:
            # Split data
            x_train, x_val, y_train, y_val = train_test_split(
                images, labels,
                test_size=self.validation_split,
                stratify=labels,
                random_state=42
            )
        else:
            x_train, y_train = images, labels
            x_val, y_val = validation_data
        
        # Process for quantum: preprocess -> extract features -> quantum prep
        logger.info("Processing images for quantum model...")
        
        # Training data
        x_train_processed = self.image_processor.preprocess_batch(x_train)
        x_train_features = self.image_processor.extract_features(x_train_processed)
        x_train_quantum = self.image_processor.prepare_for_quantum(x_train_features)
        
        # Validation data
        x_val_processed = self.image_processor.preprocess_batch(x_val)
        x_val_features = self.image_processor.extract_features(x_val_processed)
        x_val_quantum = self.image_processor.prepare_for_quantum(x_val_features)
        
        # Create datasets
        train_dataset = self._create_tf_dataset(
            x_train_quantum, y_train, is_training=True
        )
        
        val_dataset = self._create_tf_dataset(
            x_val_quantum, y_val, is_training=False
        )
        
        logger.info(f"Created quantum datasets - Features shape: {x_train_quantum.shape[1:]}")
        
        return train_dataset, val_dataset, class_weights
