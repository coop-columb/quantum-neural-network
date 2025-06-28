"""
Image preprocessing for medical imaging quantum neural network.

This module handles image loading, resizing, normalization, and
augmentation for chest X-ray images.
"""
import os
from typing import Tuple, Optional, List, Union
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging

from ..data.config import DataConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImageProcessor:
    """Preprocessor for chest X-ray images."""

    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize the image processor.

        Args:
            config: Data configuration object
        """
        self.config = config or DataConfig()
        self.scaler = StandardScaler()
        self._is_fitted = False

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        image = Image.open(image_path)

        # Convert to grayscale if needed
        if self.config.grayscale and image.mode != 'L':
            image = image.convert('L')

        # Resize to original size first (for consistency)
        image = image.resize(self.config.original_image_size, Image.LANCZOS)

        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)

        # Normalize to [0, 1]
        if self.config.normalize:
            image_array = image_array / 255.0

        return image_array

    def resize_for_quantum(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to quantum-compatible size.

        Args:
            image: Input image array

        Returns:
            Resized image
        """
        # Use TensorFlow for high-quality resizing
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        image_tensor = tf.constant(image)
        resized = tf.image.resize(
            image_tensor,
            self.config.quantum_image_size,
            method=tf.image.ResizeMethod.LANCZOS5
        )

        return resized.numpy()

    def extract_classical_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract classical features from image for hybrid processing.

        Args:
            image: Input image (original size)

        Returns:
            Feature vector of size n_classical_features
        """
        if self.config.feature_extractor == "mobilenet":
            return self._extract_mobilenet_features(image)
        elif self.config.feature_extractor == "custom":
            return self._extract_custom_features(image)
        else:
            raise ValueError(f"Unknown feature extractor: {self.config.feature_extractor}")

    def _extract_mobilenet_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features using pre-trained MobileNet."""
        # Prepare image for MobileNet
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = np.stack([image] * 3, axis=-1)

        # Resize to MobileNet input size
        image_resized = tf.image.resize(image, (224, 224))
        image_batch = tf.expand_dims(image_resized, 0)

        # Preprocess for MobileNet
        image_batch = tf.keras.applications.mobilenet_v2.preprocess_input(image_batch)

        # Load MobileNet without top layers
        if not hasattr(self, '_feature_extractor'):
            self._feature_extractor = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )
            self._feature_extractor.trainable = False

        # Extract features
        features = self._feature_extractor(image_batch, training=False)
        features = features.numpy().flatten()

        # Reduce to n_classical_features using PCA-like projection
        if len(features) > self.config.n_classical_features:
            # Simple linear projection for now
            projection_matrix = np.random.RandomState(42).randn(
                len(features), self.config.n_classical_features
            )
            projection_matrix = projection_matrix / np.linalg.norm(
                projection_matrix, axis=0, keepdims=True
            )
            features = features @ projection_matrix

        return features

    def _extract_custom_features(self, image: np.ndarray) -> np.ndarray:
        """Extract custom features optimized for chest X-rays."""
        features = []

        # 1. Global statistics
        features.extend([
            np.mean(image),
            np.std(image),
            np.median(image),
            np.percentile(image, 25),
            np.percentile(image, 75)
        ])

        # 2. Histogram features (5 bins)
        hist, _ = np.histogram(image, bins=5, range=(0, 1))
        hist = hist / hist.sum()  # Normalize
        features.extend(hist)

        # 3. Regional features (divide into 2x2 grid)
        h, w = image.shape
        regions = [
            image[:h // 2, :w // 2],  # Top-left
            image[:h // 2, w // 2:],  # Top-right
            image[h // 2:, :w // 2],  # Bottom-left
            image[h // 2:, w // 2:]  # Bottom-right
        ]

        for region in regions:
            features.extend([
                np.mean(region),
                np.std(region)
            ])

        # Ensure we have exactly n_classical_features
        features = np.array(features[:self.config.n_classical_features])
        if len(features) < self.config.n_classical_features:
            # Pad with zeros if needed
            features = np.pad(
                features,
                (0, self.config.n_classical_features - len(features)),
                'constant'
            )

        return features

    def augment_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply data augmentation to image.

        Args:
            image: Input image

        Returns:
            List of augmented images (including original)
        """
        augmented = [image]

        if not self.config.use_augmentation:
            return augmented

        # Convert to tensor for augmentation
        image_tensor = tf.constant(image)
        if len(image_tensor.shape) == 2:
            image_tensor = tf.expand_dims(image_tensor, axis=-1)

        for i in range(self.config.augmentation_factor - 1):
            aug_image = image_tensor

            # Random rotation
            if self.config.rotation_range > 0:
                angle = tf.random.uniform(
                    [], -self.config.rotation_range, self.config.rotation_range
                )
                aug_image = tf.keras.preprocessing.image.random_rotation(
                    aug_image, angle, row_axis=0, col_axis=1, channel_axis=2
                )

            # Random shift
            if self.config.width_shift_range > 0 or self.config.height_shift_range > 0:
                aug_image = tf.keras.preprocessing.image.random_shift(
                    aug_image,
                    self.config.width_shift_range,
                    self.config.height_shift_range,
                    row_axis=0, col_axis=1, channel_axis=2
                )

            # Random zoom
            if self.config.zoom_range > 0:
                zoom_factor = tf.random.uniform(
                    [], 1 - self.config.zoom_range, 1 + self.config.zoom_range
                )
                aug_image = tf.keras.preprocessing.image.random_zoom(
                    aug_image, (zoom_factor, zoom_factor),
                    row_axis=0, col_axis=1, channel_axis=2
                )

            # Horizontal flip
            if self.config.horizontal_flip and tf.random.uniform([]) > 0.5:
                aug_image = tf.image.flip_left_right(aug_image)

            # Add to list
            augmented.append(tf.squeeze(aug_image).numpy())

        return augmented

    def preprocess_batch(
            self,
            image_paths: List[str],
            labels: Optional[List[int]] = None,
            augment: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess a batch of images.

        Args:
            image_paths: List of image file paths
            labels: Optional labels for the images
            augment: Whether to apply augmentation

        Returns:
            Tuple of (quantum_images, classical_features, original_images, labels)
        """
        quantum_images = []
        classical_features = []
        original_images = []
        all_labels = [] if labels is not None else None

        for i, image_path in enumerate(image_paths):
            # Load image
            image = self.load_image(image_path)

            # Get augmented versions
            if augment:
                augmented_images = self.augment_image(image)
            else:
                augmented_images = [image]

            for aug_image in augmented_images:
                # Store original size image
                original_images.append(aug_image)

                # Extract classical features
                features = self.extract_classical_features(aug_image)
                classical_features.append(features)

                # Resize for quantum processing
                quantum_image = self.resize_for_quantum(aug_image)
                quantum_images.append(quantum_image)

                # Duplicate labels for augmented images
                if labels is not None:
                    all_labels.append(labels[i])

        # Convert to numpy arrays
        quantum_images = np.array(quantum_images)
        classical_features = np.array(classical_features)
        original_images = np.array(original_images)

        if all_labels is not None:
            all_labels = np.array(all_labels)

        # Fit scaler on classical features if needed
        if not self._is_fitted:
            self.scaler.fit(classical_features)
            self._is_fitted = True

        # Scale classical features
        classical_features = self.scaler.transform(classical_features)

        return quantum_images, classical_features, original_images, all_labels

    def save_preprocessor_state(self, filepath: str):
        """Save the preprocessor state (scaler parameters)."""
        import pickle
        state = {
            'scaler': self.scaler,
            'is_fitted': self._is_fitted,
            'config': self.config
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Saved preprocessor state to {filepath}")

    def load_preprocessor_state(self, filepath: str):
        """Load the preprocessor state."""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.scaler = state['scaler']
        self._is_fitted = state['is_fitted']
        self.config = state['config']
        logger.info(f"Loaded preprocessor state from {filepath}")