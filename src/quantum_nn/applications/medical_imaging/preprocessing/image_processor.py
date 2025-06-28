"""
Image preprocessing for medical imaging quantum neural network.

This module handles image loading, resizing, normalization, and
feature extraction for medical images.
"""

from typing import Tuple, Optional, List, Union
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image processor for medical imaging applications.

    This class handles image preprocessing including resizing, normalization,
    feature extraction using pretrained CNNs, and preparation for quantum processing.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        quantum_target_size: Tuple[int, int] = (28, 28),
        normalize: bool = True,
        augment: bool = False,
    ):
        """
        Initialize the image processor.

        Args:
            target_size: Target size for classical processing
            quantum_target_size: Target size for quantum processing
            normalize: Whether to normalize images
            augment: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.quantum_target_size = quantum_target_size
        self.normalize = normalize
        self.augment = augment

        # Initialize feature extractor (MobileNet)
        self._init_feature_extractor()

        logger.info(
            f"ImageProcessor initialized with target_size={target_size}, "
            f"quantum_target_size={quantum_target_size}"
        )

    def _init_feature_extractor(self):
        """Initialize the feature extraction model."""
        try:
            # Load MobileNetV2 for feature extraction
            self.feature_extractor = tf.keras.applications.MobileNetV2(
                input_shape=(*self.target_size, 3),
                include_top=False,
                weights="imagenet",
                pooling="avg",
            )
            self.feature_extractor.trainable = False
            logger.info("MobileNetV2 feature extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to load MobileNetV2: {e}. Using simple CNN.")
            # Fallback to simple CNN
            self.feature_extractor = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(*self.target_size, 3)),
                    tf.keras.layers.Conv2D(32, 3, activation="relu"),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(128, activation="relu"),
                ]
            )

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image.

        Args:
            image: Input image array

        Returns:
            Preprocessed image
        """
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            image = tf.constant(image, dtype=tf.float32)

        # Resize image
        image = tf.image.resize(image, self.target_size)

        # Normalize to [0, 1]
        if self.normalize:
            image = tf.cast(image, tf.float32) / 255.0

        # Apply augmentation if requested (only during training)
        if self.augment:
            image = self._augment_image(image)

        return image.numpy()

    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess a batch of images.

        Args:
            images: Batch of input images

        Returns:
            Preprocessed image batch
        """
        processed_images = []

        for image in images:
            processed_image = self.preprocess_image(image)
            processed_images.append(processed_image)

        return np.array(processed_images)

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features using pretrained CNN.

        Args:
            images: Preprocessed images

        Returns:
            Extracted features
        """
        # Ensure images are in correct format
        if images.dtype != np.float32:
            images = images.astype(np.float32)

        # Extract features using feature extractor
        features = self.feature_extractor(images)

        return features.numpy()

    def prepare_for_quantum(self, features: np.ndarray) -> np.ndarray:
        """
        Prepare features for quantum processing.

        Args:
            features: CNN features

        Returns:
            Quantum-ready features
        """
        # For quantum processing, we need to reduce dimensionality
        # and ensure proper normalization

        # Use PCA-like dimensionality reduction to quantum target size
        target_dim = self.quantum_target_size[0] * self.quantum_target_size[1] * 3

        if features.shape[-1] > target_dim:
            # Simple linear projection to reduce dimensions
            projection_matrix = np.random.normal(0, 1, (features.shape[-1], target_dim))
            features = np.dot(features, projection_matrix)
        elif features.shape[-1] < target_dim:
            # Pad with zeros if needed
            padding = target_dim - features.shape[-1]
            features = np.pad(features, ((0, 0), (0, padding)), mode="constant")

        # Normalize to [-1, 1] range for quantum encoding
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        features = np.tanh(features)  # Ensure bounded range

        return features

    def _augment_image(self, image: tf.Tensor) -> tf.Tensor:
        """Apply data augmentation to image."""
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Random rotation
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))

        # Random brightness and contrast
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)

        # Ensure values stay in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image
