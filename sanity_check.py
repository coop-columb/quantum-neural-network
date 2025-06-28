#!/usr/bin/env python3
"""
Sanity check script for Quantum Neural Network Medical Imaging project.

This script verifies all components are properly set up.
"""
import os
import sys
import subprocess
from pathlib import Path


def check_environment():
    """Check Python environment setup."""
    print("1. ENVIRONMENT CHECK")
    print("=" * 50)

    # Check Python version
    python_version = sys.version.split()[0]
    print(f"✓ Python version: {python_version}")

    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    venv_path = os.environ.get('VIRTUAL_ENV', 'Not detected')
    print(f"✓ Virtual environment: {'Active' if in_venv else 'Not active'} ({venv_path})")

    # Check current directory
    cwd = os.getcwd()
    print(f"✓ Current directory: {cwd}")

    print()


def check_dependencies():
    """Check if required packages are installed."""
    print("2. DEPENDENCIES CHECK")
    print("=" * 50)

    required_packages = [
        "tensorflow",
        "pennylane",
        "qiskit",
        "numpy",
        "matplotlib",
        "pandas",
        "scikit-learn"
    ]

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package}: Installed")
        except ImportError:
            print(f"✗ {package}: NOT INSTALLED")

    print()


def check_project_structure():
    """Check if project directories and files exist."""
    print("3. PROJECT STRUCTURE CHECK")
    print("=" * 50)

    # Check main directories
    paths_to_check = [
        "src/quantum_nn",
        "src/quantum_nn/applications",
        "src/quantum_nn/applications/medical_imaging",
        "src/quantum_nn/applications/medical_imaging/data",
        "src/quantum_nn/applications/medical_imaging/models",
        "src/quantum_nn/applications/medical_imaging/utils",
        "src/quantum_nn/applications/medical_imaging/preprocessing",
        "src/quantum_nn/applications/medical_imaging/evaluation"
    ]

    for path in paths_to_check:
        exists = os.path.exists(path)
        print(f"{'✓' if exists else '✗'} {path}: {'Exists' if exists else 'MISSING'}")

    print()


def check_created_files():
    """Check if our created files exist."""
    print("4. CREATED FILES CHECK")
    print("=" * 50)

    files_to_check = [
        ("Configuration", "src/quantum_nn/applications/medical_imaging/data/config.py"),
        ("Downloader", "src/quantum_nn/applications/medical_imaging/data/downloader.py"),
        ("Test Setup", "src/quantum_nn/applications/medical_imaging/test_setup.py"),
        ("Planning Doc", "docs/applications/medical_image_classification_plan.md")
    ]

    for name, filepath in files_to_check:
        exists = os.path.exists(filepath)
        if exists:
            size = os.path.getsize(filepath)
            print(f"✓ {name}: Exists ({size} bytes)")
        else:
            print(f"✗ {name}: MISSING")

    print()


def check_data_directories():
    """Check if data directories were created."""
    print("5. DATA DIRECTORIES CHECK")
    print("=" * 50)

    data_dirs = [
        "data/medical_imaging",
        "data/medical_imaging/raw",
        "data/medical_imaging/processed",
        "data/medical_imaging/cache"
    ]

    for dir_path in data_dirs:
        exists = os.path.exists(dir_path)
        print(f"{'✓' if exists else '✗'} {dir_path}: {'Exists' if exists else 'MISSING'}")

    # Check for dataset
    dataset_path = "data/medical_imaging/raw/chest_xray"
    if os.path.exists(dataset_path):
        print(f"\n✓ DATASET FOUND at {dataset_path}")
        # Count images
        total_images = sum(1 for _ in Path(dataset_path).rglob("*.jpeg"))
        print(f"  Total images: {total_images}")
    else:
        print(f"\n✗ Dataset not yet downloaded")
        print(f"  Expected location: {dataset_path}")

    print()


def check_git_status():
    """Check git repository status."""
    print("6. GIT STATUS CHECK")
    print("=" * 50)

    try:
        # Get current branch
        branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
        print(f"✓ Current branch: {branch}")

        # Check for uncommitted changes
        status = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
        if status:
            print(f"⚠ Uncommitted changes detected:")
            print(status)
        else:
            print("✓ Working tree clean")
    except subprocess.CalledProcessError:
        print("✗ Not a git repository or git error")

    print()


def main():
    """Run all sanity checks."""
    print("\n" + "=" * 60)
    print("QUANTUM NEURAL NETWORK - MEDICAL IMAGING")
    print("PROJECT SANITY CHECK")
    print("=" * 60 + "\n")

    check_environment()
    check_dependencies()
    check_project_structure()
    check_created_files()
    check_data_directories()
    check_git_status()

    print("=" * 60)
    print("SANITY CHECK COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download the chest X-ray dataset from Kaggle")
    print("2. Create preprocessing pipeline")
    print("3. Implement quantum neural network model")
    print("4. Train and evaluate the model")


if __name__ == "__main__":
    main()