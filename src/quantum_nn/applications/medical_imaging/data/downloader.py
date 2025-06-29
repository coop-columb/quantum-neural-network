"""
Dataset downloader for medical imaging data.

This module handles downloading and extracting the chest X-ray dataset.
Since the dataset is hosted on Kaggle, it provides instructions for manual
download if automatic download is not possible.
"""

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from ..data.config import DataConfig


class DatasetDownloader:
    """Handles downloading and extracting medical imaging datasets."""

    def __init__(self, config: DataConfig):
        """
        Initialize the dataset downloader.

        Args:
            config: Data configuration object
        """
        self.config = config
        self.dataset_path = Path(config.raw_data_dir) / "chest_xray"
        self.zip_path = Path(config.raw_data_dir) / "chest_xray.zip"

    def download_dataset(
        self,
        kaggle_json_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """
        Download the chest X-ray dataset.

        Args:
            kaggle_json_path: Path to kaggle.json authentication file
            progress_callback: Callback for download progress (bytes_downloaded, total_bytes)

        Returns:
            True if download successful, False otherwise
        """
        # Check if dataset already exists
        if self.dataset_path.exists() and self._verify_dataset():
            print(f"Dataset already exists at {self.dataset_path}")
            return True

        # Create directories
        self.config.create_directories()

        # Try automatic download if kaggle credentials provided
        if kaggle_json_path and os.path.exists(kaggle_json_path):
            return self._download_with_kaggle(kaggle_json_path, progress_callback)
        else:
            # Provide manual download instructions
            self._print_manual_instructions()
            return False

    def _download_with_kaggle(
        self,
        kaggle_json_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """Download dataset using Kaggle API."""
        try:
            # Set up kaggle credentials
            import kaggle

            os.environ["KAGGLE_CONFIG_DIR"] = os.path.dirname(kaggle_json_path)

            # Download dataset
            print("Downloading dataset from Kaggle...")
            kaggle.api.dataset_download_files(
                "paultimothymooney/chest-xray-pneumonia",
                path=self.config.raw_data_dir,
                unzip=True,
            )

            print("Dataset downloaded successfully!")
            return True

        except ImportError:
            print(
                "Kaggle package not installed. Please install with: pip install kaggle"
            )
            return False
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False

    def _print_manual_instructions(self):
        """Print instructions for manual dataset download."""
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        print(
            "\n1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
        )
        print("2. Sign in to Kaggle (create free account if needed)")
        print("3. Click 'Download' button to download the dataset")
        print("4. Extract the downloaded archive")
        print(
            f"5. Place the extracted 'chest_xray' folder in: {self.config.raw_data_dir}"
        )
        print("\nExpected structure after extraction:")
        print("  chest_xray/")
        print("    ├── train/")
        print("    │   ├── NORMAL/")
        print("    │   └── PNEUMONIA/")
        print("    ├── test/")
        print("    │   ├── NORMAL/")
        print("    │   └── PNEUMONIA/")
        print("    └── val/")
        print("        ├── NORMAL/")
        print("        └── PNEUMONIA/")
        print("\n" + "=" * 60)

    def extract_dataset(self, zip_path: Optional[str] = None) -> bool:
        """
        Extract dataset from zip file.

        Args:
            zip_path: Path to zip file (uses default if None)

        Returns:
            True if extraction successful
        """
        zip_path = Path(zip_path) if zip_path else self.zip_path

        if not zip_path.exists():
            print(f"Zip file not found: {zip_path}")
            return False

        print(f"Extracting dataset from {zip_path}...")

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.config.raw_data_dir)

            print("Extraction complete!")

            # Remove zip file to save space
            if self.zip_path.exists():
                os.remove(self.zip_path)

            return True

        except Exception as e:
            print(f"Error extracting dataset: {e}")
            return False

    def _verify_dataset(self) -> bool:
        """Verify that the dataset structure is correct."""
        expected_dirs = [
            self.dataset_path / "train" / "NORMAL",
            self.dataset_path / "train" / "PNEUMONIA",
            self.dataset_path / "test" / "NORMAL",
            self.dataset_path / "test" / "PNEUMONIA",
            self.dataset_path / "val" / "NORMAL",
            self.dataset_path / "val" / "PNEUMONIA",
        ]

        for dir_path in expected_dirs:
            if not dir_path.exists():
                return False

            # Check if directories contain images
            image_files = list(dir_path.glob("*.jpeg")) + list(dir_path.glob("*.jpg"))
            if len(image_files) == 0:
                return False

        return True

    def get_dataset_stats(self) -> dict:
        """Get statistics about the downloaded dataset."""
        if not self._verify_dataset():
            return {"error": "Dataset not found or incomplete"}

        stats = {"dataset_path": str(self.dataset_path), "splits": {}}

        for split in ["train", "test", "val"]:
            split_path = self.dataset_path / split
            normal_count = len(list((split_path / "NORMAL").glob("*.jpeg")))
            pneumonia_count = len(list((split_path / "PNEUMONIA").glob("*.jpeg")))

            stats["splits"][split] = {
                "normal": normal_count,
                "pneumonia": pneumonia_count,
                "total": normal_count + pneumonia_count,
            }

        stats["total_images"] = sum(
            split_stats["total"] for split_stats in stats["splits"].values()
        )

        return stats

    def create_sample_dataset(self, n_samples_per_class: int = 10) -> bool:
        """
        Create a small sample dataset for testing.

        Args:
            n_samples_per_class: Number of samples per class to include

        Returns:
            True if sample dataset created successfully
        """
        if not self._verify_dataset():
            print("Full dataset not found. Cannot create sample dataset.")
            return False

        sample_dir = Path(self.config.raw_data_dir) / "chest_xray_sample"

        # Create sample directory structure
        for split in ["train", "test", "val"]:
            for class_name in ["NORMAL", "PNEUMONIA"]:
                sample_split_dir = sample_dir / split / class_name
                sample_split_dir.mkdir(parents=True, exist_ok=True)

                # Copy sample images
                source_dir = self.dataset_path / split / class_name
                image_files = list(source_dir.glob("*.jpeg"))[:n_samples_per_class]

                for img_file in image_files:
                    shutil.copy2(img_file, sample_split_dir / img_file.name)

        print(f"Sample dataset created at {sample_dir}")
        return True


def download_chest_xray_dataset(
    config: Optional[DataConfig] = None, kaggle_json_path: Optional[str] = None
) -> bool:
    """
    Convenience function to download the chest X-ray dataset.

    Args:
        config: Data configuration (uses default if None)
        kaggle_json_path: Path to kaggle.json file

    Returns:
        True if dataset is ready to use
    """
    config = config or DataConfig()
    downloader = DatasetDownloader(config)

    return downloader.download_dataset(kaggle_json_path)
