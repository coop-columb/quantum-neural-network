"""
Test script to verify medical imaging application setup.

This script tests the configuration and data downloader modules
to ensure everything is set up correctly.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from quantum_nn.applications.medical_imaging.data.config import DataConfig, default_config
from quantum_nn.applications.medical_imaging.data.downloader import DatasetDownloader


def test_configuration():
    """Test the data configuration."""
    print("=" * 60)
    print("TESTING DATA CONFIGURATION")
    print("=" * 60)

    # Create config instance
    config = DataConfig()

    # Print configuration details
    print("\nConfiguration Details:")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Base directory: {config.base_data_dir}")
    print(f"  Original image size: {config.original_image_size}")
    print(f"  Quantum image size: {config.quantum_image_size}")
    print(f"  Number of qubits: {config.n_qubits}")
    print(f"  Classical features: {config.n_classical_features}")

    # Test directory creation
    print("\nCreating data directories...")
    config.create_directories()

    # Verify directories were created
    dirs_created = all([
        os.path.exists(config.raw_data_dir),
        os.path.exists(config.processed_data_dir),
        os.path.exists(config.cache_dir)
    ])

    if dirs_created:
        print("✓ All directories created successfully!")
    else:
        print("✗ Error creating directories")

    # Print data statistics
    print("\nData Statistics:")
    stats = config.get_data_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return config


def test_downloader(config: DataConfig):
    """Test the dataset downloader."""
    print("\n" + "=" * 60)
    print("TESTING DATASET DOWNLOADER")
    print("=" * 60)

    # Create downloader
    downloader = DatasetDownloader(config)

    # Check if dataset already exists
    dataset_path = Path(config.raw_data_dir) / "chest_xray"
    if dataset_path.exists():
        print(f"\n✓ Dataset found at: {dataset_path}")

        # Get dataset statistics
        stats = downloader.get_dataset_stats()
        if "error" not in stats:
            print("\nDataset Statistics:")
            print(f"  Total images: {stats['total_images']}")
            for split, split_stats in stats['splits'].items():
                print(f"\n  {split.upper()} set:")
                print(f"    Normal: {split_stats['normal']}")
                print(f"    Pneumonia: {split_stats['pneumonia']}")
                print(f"    Total: {split_stats['total']}")
        else:
            print(f"\n✗ {stats['error']}")
    else:
        print(f"\n✗ Dataset not found at: {dataset_path}")
        print("\nTo download the dataset:")

        # Check for kaggle credentials
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            print("  - Kaggle credentials found. Run:")
            print("    downloader.download_dataset(kaggle_json_path='~/.kaggle/kaggle.json')")
        else:
            print("  - No Kaggle credentials found.")
            downloader._print_manual_instructions()

    return downloader


def create_sample_dataset(downloader: DatasetDownloader):
    """Create a small sample dataset for testing."""
    print("\n" + "=" * 60)
    print("CREATING SAMPLE DATASET")
    print("=" * 60)

    response = input("\nCreate a sample dataset with 10 images per class? (y/n): ")
    if response.lower() == 'y':
        success = downloader.create_sample_dataset(n_samples_per_class=10)
        if success:
            print("✓ Sample dataset created successfully!")
        else:
            print("✗ Could not create sample dataset")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("QUANTUM NEURAL NETWORK - MEDICAL IMAGING APPLICATION")
    print("Setup Verification Script")
    print("=" * 60)

    # Test configuration
    config = test_configuration()

    # Test downloader
    downloader = test_downloader(config)

    # Optionally create sample dataset
    if downloader._verify_dataset():
        create_sample_dataset(downloader)

    print("\n" + "=" * 60)
    print("SETUP VERIFICATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Ensure the chest X-ray dataset is downloaded")
    print("2. Run the preprocessing pipeline")
    print("3. Train the quantum neural network model")


if __name__ == "__main__":
    main()