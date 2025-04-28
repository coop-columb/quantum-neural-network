"""
Tests for the benchmark runner.
"""
import os
import shutil
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from quantum_nn.benchmarks import BenchmarkRunner
from quantum_nn.models import QuantumModel


class TestBenchmarkRunner:
    """Test suite for benchmark runner."""

    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test benchmark runner initialization."""
        runner = BenchmarkRunner(output_dir=self.temp_dir)
        
        assert runner.output_dir == self.temp_dir
        assert runner.metrics == ['loss', 'accuracy']
        assert runner.verbose == 1
        
        # Test with custom metrics
        custom_metrics = ['loss', 'mean_absolute_error']
        runner = BenchmarkRunner(
            output_dir=self.temp_dir,
            metrics=custom_metrics,
            verbose=0
        )
        
        assert runner.metrics == custom_metrics
        assert runner.verbose == 0
    
    def test_run_benchmark(self):
        """Test running a benchmark."""
        # Create a mock QuantumModel
        mock_model = MockQuantumModel()
        
        # Create simple test data
        x_train = np.random.random((10, 2))
        y_train = np.random.randint(0, 2, (10, 1)).astype(np.float32)
        x_test = np.random.random((5, 2))
        y_test = np.random.randint(0, 2, (5, 1)).astype(np.float32)
        
        # Create benchmark runner
        runner = BenchmarkRunner(output_dir=self.temp_dir, verbose=0)
        
        # Run benchmark
        results = runner.run_benchmark(
            model=mock_model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            task_name="test_task",
            model_name="test_model",
            epochs=2,
            batch_size=5
        )
        
        # Check results
        assert isinstance(results, dict)
        assert results["task_name"] == "test_task"
        assert results["model_name"] == "test_model"
        assert "performance" in results
        assert "timing" in results
        assert "history" in results
        
        # Check that files were created
        result_dirs = os.listdir(self.temp_dir)
        assert len(result_dirs) == 1
        
        result_dir = os.path.join(self.temp_dir, result_dirs[0])
        assert os.path.isfile(os.path.join(result_dir, "results.json"))
        assert os.path.isfile(os.path.join(result_dir, "training_history.png"))
        assert os.path.isfile(os.path.join(result_dir, "performance_summary.png"))
    
    def test_compare_models(self):
        """Test comparing multiple models."""
        # Create mock models
        models = {
            "model_1": MockQuantumModel(),
            "model_2": MockQuantumModel()
        }
        
        # Create simple test data
        x_train = np.random.random((10, 2))
        y_train = np.random.randint(0, 2, (10, 1)).astype(np.float32)
        x_test = np.random.random((5, 2))
        y_test = np.random.randint(0, 2, (5, 1)).astype(np.float32)
        
        # Create benchmark runner
        runner = BenchmarkRunner(output_dir=self.temp_dir, verbose=0)
        
        # Run comparison
        results = runner.compare_models(
            models=models,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            task_name="test_task",
            epochs=2,
            batch_size=5
        )
        
        # Check results
        assert isinstance(results, dict)
        assert len(results) == 2
        assert "model_1" in results
        assert "model_2" in results
        
        # Check that comparison directory was created
        dirs = [d for d in os.listdir(self.temp_dir) if d.startswith("comparison_")]
        assert len(dirs) == 1
        
        comparison_dir = os.path.join(self.temp_dir, dirs[0])
        assert os.path.isfile(os.path.join(comparison_dir, "comparison_results.json"))
        assert os.path.isfile(os.path.join(comparison_dir, "performance_comparison.png"))
        assert os.path.isfile(os.path.join(comparison_dir, "timing_comparison.png"))


class MockQuantumModel(QuantumModel):
    """Mock implementation of QuantumModel for testing."""
    
    def __init__(self):
        """Initialize mock model."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
    
    def fit(self, x, y, **kwargs):
        """Mock fit method."""
        # Return a mock history object with random metrics
        history = {
            "loss": [0.5, 0.4],
            "accuracy": [0.6, 0.7],
            "val_loss": [0.55, 0.45],
            "val_accuracy": [0.55, 0.65]
        }
        return type('obj', (object,), {'history': history})
    
    def evaluate(self, x, y, **kwargs):
        """Mock evaluate method."""
        return [0.4, 0.7]
    
    def predict(self, x, **kwargs):
        """Mock predict method."""
        return np.random.random((len(x), 1))
