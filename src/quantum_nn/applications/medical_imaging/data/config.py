"""
Configuration for medical imaging dataset.

This module contains all configuration parameters for the chest X-ray
pneumonia detection dataset and preprocessing settings.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DataConfig:
    """Configuration for medical imaging data pipeline."""

    # Dataset information
    dataset_name: str = "chest_xray_pneumonia"
    dataset_url: str = (
        "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
    )

    # Data directories
    base_data_dir: str = "./data/medical_imaging"
    raw_data_dir: str = os.path.join(base_data_dir, "raw")
    processed_data_dir: str = os.path.join(base_data_dir, "processed")
    cache_dir: str = os.path.join(base_data_dir, "cache")

    # Image parameters
    original_image_size: Tuple[int, int] = (224, 224)
    quantum_image_size: Tuple[int, int] = (28, 28)  # Reduced for quantum processing
    grayscale: bool = True
    normalize: bool = True

    # Classical feature extraction
    n_classical_features: int = 16  # Features to extract before quantum processing
    feature_extractor: str = "mobilenet"  # Options: "mobilenet", "resnet", "custom"

    # Data split parameters
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

    # Augmentation parameters
    use_augmentation: bool = True
    augmentation_factor: int = 2  # How many augmented versions per image
    rotation_range: float = 20.0
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    horizontal_flip: bool = True
    zoom_range: float = 0.2

    # Batch processing
    batch_size: int = 16
    prefetch_buffer: int = 2

    # Quantum encoding parameters
    encoding_method: str = "amplitude"  # Options: "amplitude", "angle", "hybrid"
    n_qubits: int = 4

    # Labels
    class_names: Tuple[str, str] = ("NORMAL", "PNEUMONIA")
    n_classes: int = 2

    # Performance parameters
    num_workers: int = 4
    max_dataset_size: Optional[int] = None  # Limit dataset size for development

    def create_directories(self):
        """Create all necessary directories for data storage."""
        os.makedirs(self.base_data_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Create subdirectories for train/val/test splits
        for split in ["train", "val", "test"]:
            for class_name in self.class_names:
                os.makedirs(
                    os.path.join(self.processed_data_dir, split, class_name),
                    exist_ok=True,
                )

    def get_data_stats(self) -> dict:
        """Get statistics about the dataset configuration."""
        return {
            "dataset": self.dataset_name,
            "image_size": {
                "original": self.original_image_size,
                "quantum": self.quantum_image_size,
                "reduction_factor": self.original_image_size[0]
                / self.quantum_image_size[0],
            },
            "features": {
                "classical": self.n_classical_features,
                "quantum_qubits": self.n_qubits,
                "encoding": self.encoding_method,
            },
            "augmentation": {
                "enabled": self.use_augmentation,
                "factor": self.augmentation_factor if self.use_augmentation else 1,
            },
            "splits": {
                "train": self.train_split,
                "val": self.val_split,
                "test": self.test_split,
            },
        }


# Default configuration instance
default_config = DataConfig()
