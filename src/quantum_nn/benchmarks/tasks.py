"""
Benchmark tasks for quantum neural networks.

This module provides standardized tasks for benchmarking
quantum neural network performance.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import make_classification, make_regression, make_moons, load_digits


def get_benchmark_task(
    task_name: str,
    n_samples: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
    task_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Get a standardized benchmark task.
    
    Args:
        task_name: Name of the task
        n_samples: Number of samples to generate
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        task_kwargs: Additional task-specific parameters
        
    Returns:
        Tuple of (x_train, y_train, x_test, y_test, task_info)
    """
    task_kwargs = task_kwargs or {}
    
    # Dispatch to appropriate task generator
    if task_name == "binary_classification":
        data = _generate_binary_classification(n_samples, random_state, **task_kwargs)
    elif task_name == "multiclass_classification":
        data = _generate_multiclass_classification(n_samples, random_state, **task_kwargs)
    elif task_name == "regression":
        data = _generate_regression(n_samples, random_state, **task_kwargs)
    elif task_name == "moons":
        data = _generate_moons(n_samples, random_state, **task_kwargs)
    elif task_name == "digits":
        data = _generate_digits(random_state, **task_kwargs)
    elif task_name == "quantum_simulation":
        data = _generate_quantum_simulation(n_samples, random_state, **task_kwargs)
    elif task_name == "entanglement_classification":
        data = _generate_entanglement_classification(n_samples, random_state, **task_kwargs)
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Unpack data
    X, y, task_info = data
    
    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return x_train, y_train, x_test, y_test, task_info


def _generate_binary_classification(
    n_samples: int, random_state: int, n_features: int = 10, **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a binary classification task."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=kwargs.get("n_informative", 5),
        n_redundant=kwargs.get("n_redundant", 2),
        n_classes=2,
        random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape target to (n_samples, 1)
    y = y.reshape(-1, 1)
    
    task_info = {
        "task_type": "classification",
        "n_classes": 2,
        "n_features": n_features,
        "feature_names": [f"feature_{i}" for i in range(n_features)],
        "class_names": ["class_0", "class_1"]
    }
    
    return X, y, task_info


def _generate_multiclass_classification(
    n_samples: int, random_state: int, n_features: int = 10, n_classes: int = 4, **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a multiclass classification task."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=kwargs.get("n_informative", 5),
        n_redundant=kwargs.get("n_redundant", 2),
        n_classes=n_classes,
        random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # One-hot encode targets
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))
    
    task_info = {
        "task_type": "classification",
        "n_classes": n_classes,
        "n_features": n_features,
        "feature_names": [f"feature_{i}" for i in range(n_features)],
        "class_names": [f"class_{i}" for i in range(n_classes)]
    }
    
    return X, y, task_info


def _generate_regression(
    n_samples: int, random_state: int, n_features: int = 10, **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a regression task."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=kwargs.get("n_informative", 5),
        noise=kwargs.get("noise", 0.1),
        random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Scale targets
    y = y.reshape(-1, 1)
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y)
    
    task_info = {
        "task_type": "regression",
        "n_features": n_features,
        "feature_names": [f"feature_{i}" for i in range(n_features)],
        "target_name": "target",
        "y_scaler": y_scaler
    }
    
    return X, y, task_info


def _generate_moons(
    n_samples: int, random_state: int, noise: float = 0.1, **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate the moons dataset."""
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape target to (n_samples, 1)
    y = y.reshape(-1, 1)
    
    task_info = {
        "task_type": "classification",
        "n_classes": 2,
        "n_features": 2,
        "feature_names": ["x", "y"],
        "class_names": ["moon_1", "moon_2"],
        "description": "Two interleaving half moons"
    }
    
    return X, y, task_info


def _generate_digits(
    random_state: int, **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate the digits dataset."""
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # One-hot encode targets
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))
    
    task_info = {
        "task_type": "classification",
        "n_classes": 10,
        "n_features": X.shape[1],
        "feature_names": [f"pixel_{i}" for i in range(X.shape[1])],
        "class_names": [str(i) for i in range(10)],
        "description": "Handwritten digit recognition",
        "image_shape": (8, 8)
    }
    
    return X, y, task_info


def _generate_quantum_simulation(
    n_samples: int, random_state: int, n_qubits: int = 4, **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate a quantum simulation task.
    
    This task involves predicting the output of a parameterized quantum circuit.
    """
    np.random.seed(random_state)
    
    # Generate random parameters for quantum circuits
    X = np.random.uniform(-np.pi, np.pi, size=(n_samples, 3 * n_qubits))
    
    # Simulate the output of a quantum circuit
    # For simplicity, we use a classical approximation here
    y = np.zeros((n_samples, n_qubits))
    
    for i in range(n_samples):
        params = X[i]
        
        # Simulate a simplified quantum circuit output
        # Each qubit's output is influenced by its parameters
        for j in range(n_qubits):
            # Extract parameters for this qubit
            rx = params[j * 3]
            ry = params[j * 3 + 1]
            rz = params[j * 3 + 2]
            
            # Simple classical approximation of quantum rotation
            y[i, j] = np.cos(rx) * np.cos(ry) * np.cos(rz) + np.sin(rx) * np.sin(ry) * np.sin(rz)
    
    task_info = {
        "task_type": "regression",
        "n_features": X.shape[1],
        "n_outputs": y.shape[1],
        "feature_names": [f"param_{i}" for i in range(X.shape[1])],
        "target_names": [f"qubit_{i}" for i in range(n_qubits)],
        "description": "Quantum circuit simulation",
        "n_qubits": n_qubits
    }
    
    return X, y, task_info


def _generate_entanglement_classification(
    n_samples: int, random_state: int, n_qubits: int = 4, **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate an entanglement classification task.
    
    This task involves classifying whether a quantum state is entangled or not.
    """
    np.random.seed(random_state)
    
    # Generate random parameters for quantum circuits
    X = np.random.uniform(-np.pi, np.pi, size=(n_samples, 3 * n_qubits))
    
    # Add an explicit entangling parameter
    entangling_strength = np.random.uniform(0, np.pi, size=(n_samples, 1))
    X = np.hstack([X, entangling_strength])
    
    # Generate labels: 1 if entangled, 0 if not
    # We use a threshold on the entangling parameter as a simplification
    threshold = np.pi / 4
    y = (entangling_strength > threshold).astype(np.float32)
    
    task_info = {
        "task_type": "classification",
        "n_classes": 2,
        "n_features": X.shape[1],
        "feature_names": [f"param_{i}" for i in range(X.shape[1] - 1)] + ["entangling_param"],
        "class_names": ["separable", "entangled"],
        "description": "Entanglement classification",
        "n_qubits": n_qubits
    }
    
    return X, y, task_info
