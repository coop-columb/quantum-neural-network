"""
Tests for benchmark tasks.
"""
import numpy as np
import pytest

from quantum_nn.benchmarks import get_benchmark_task


class TestBenchmarkTasks:
    """Test suite for benchmark tasks."""

    def test_binary_classification(self):
        """Test binary classification task."""
        x_train, y_train, x_test, y_test, task_info = get_benchmark_task(
            task_name="binary_classification",
            n_samples=100,
            test_size=0.2,
            random_state=42
        )
        
        # Check shapes
        assert x_train.shape[0] == 80
        assert y_train.shape[0] == 80
        assert x_test.shape[0] == 20
        assert y_test.shape[0] == 20
        
        assert y_train.shape[1] == 1
        assert y_test.shape[1] == 1
        
        # Check task info
        assert task_info["task_type"] == "classification"
        assert task_info["n_classes"] == 2
    
    def test_multiclass_classification(self):
        """Test multiclass classification task."""
        x_train, y_train, x_test, y_test, task_info = get_benchmark_task(
            task_name="multiclass_classification",
            n_samples=100,
            test_size=0.2,
            random_state=42,
            task_kwargs={"n_classes": 3}
        )
        
        # Check shapes
        assert x_train.shape[0] == 80
        assert y_train.shape[0] == 80
        assert x_test.shape[0] == 20
        assert y_test.shape[0] == 20
        
        assert y_train.shape[1] == 3  # One-hot encoded
        assert y_test.shape[1] == 3
        
        # Check task info
        assert task_info["task_type"] == "classification"
        assert task_info["n_classes"] == 3
    
    def test_regression(self):
        """Test regression task."""
        x_train, y_train, x_test, y_test, task_info = get_benchmark_task(
            task_name="regression",
            n_samples=100,
            test_size=0.2,
            random_state=42
        )
        
        # Check shapes
        assert x_train.shape[0] == 80
        assert y_train.shape[0] == 80
        assert x_test.shape[0] == 20
        assert y_test.shape[0] == 20
        
        assert y_train.shape[1] == 1
        assert y_test.shape[1] == 1
        
        # Check task info
        assert task_info["task_type"] == "regression"
    
    def test_moons(self):
        """Test moons dataset."""
        x_train, y_train, x_test, y_test, task_info = get_benchmark_task(
            task_name="moons",
            n_samples=100,
            test_size=0.2,
            random_state=42
        )
        
        # Check shapes
        assert x_train.shape[0] == 80
        assert y_train.shape[0] == 80
        assert x_test.shape[0] == 20
        assert y_test.shape[0] == 20
        
        assert x_train.shape[1] == 2
        assert y_train.shape[1] == 1
        
        # Check task info
        assert task_info["task_type"] == "classification"
        assert task_info["n_classes"] == 2
    
    def test_quantum_simulation(self):
        """Test quantum simulation task."""
        x_train, y_train, x_test, y_test, task_info = get_benchmark_task(
            task_name="quantum_simulation",
            n_samples=100,
            test_size=0.2,
            random_state=42,
            task_kwargs={"n_qubits": 3}
        )
        
        # Check shapes
        assert x_train.shape[0] == 80
        assert y_train.shape[0] == 80
        assert x_test.shape[0] == 20
        assert y_test.shape[0] == 20
        
        assert x_train.shape[1] == 9  # 3 parameters per qubit
        assert y_train.shape[1] == 3  # One output per qubit
        
        # Check task info
        assert task_info["task_type"] == "regression"
        assert task_info["n_qubits"] == 3
    
    def test_entanglement_classification(self):
        """Test entanglement classification task."""
        x_train, y_train, x_test, y_test, task_info = get_benchmark_task(
            task_name="entanglement_classification",
            n_samples=100,
            test_size=0.2,
            random_state=42
        )
        
        # Check shapes
        assert x_train.shape[0] == 80
        assert y_train.shape[0] == 80
        assert x_test.shape[0] == 20
        assert y_test.shape[0] == 20
        
        assert y_train.shape[1] == 1
        assert y_test.shape[1] == 1
        
        # Check task info
        assert task_info["task_type"] == "classification"
        assert task_info["n_classes"] == 2
    
    def test_invalid_task(self):
        """Test error for invalid task."""
        with pytest.raises(ValueError):
            get_benchmark_task(task_name="invalid_task")
