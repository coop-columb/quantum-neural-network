"""
Benchmark runner for quantum neural networks.

This module provides a framework for running standardized
benchmarks on quantum neural network models.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from quantum_nn.models import QuantumModel
from quantum_nn.visualization import plot_training_trajectory


class BenchmarkRunner:
    """
    A framework for running standardized benchmarks on quantum neural networks.

    This class handles the execution, measurement, and reporting of benchmarks
    for quantum neural network models.
    """

    def __init__(
        self,
        output_dir: str = "./experiments/results",
        metrics: Optional[List[str]] = None,
        verbose: int = 1,
    ):
        """
        Initialize a benchmark runner.

        Args:
            output_dir: Directory to save benchmark results
            metrics: Metrics to track (defaults to ['loss', 'accuracy'])
            verbose: Verbosity level (0: silent, 1: progress, 2: detailed)
        """
        self.output_dir = output_dir
        self.metrics = metrics or ["loss", "accuracy"]
        self.verbose = verbose

        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def run_benchmark(
        self,
        model: QuantumModel,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        task_name: str,
        model_name: str,
        epochs: int = 10,
        batch_size: int = 16,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> Dict[str, Any]:
        """
        Run a benchmark on a quantum neural network model.

        Args:
            model: Quantum neural network model to benchmark
            x_train: Training input data
            y_train: Training target data
            x_test: Test input data
            y_test: Test target data
            task_name: Name of the benchmark task
            model_name: Name of the model being benchmarked
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: Keras callbacks for training

        Returns:
            Dictionary containing benchmark results
        """
        if self.verbose >= 1:
            print(f"Running benchmark: {task_name} with model: {model_name}")

        # Generate timestamp for this benchmark run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{task_name}_{model_name}_{timestamp}"

        # Create result directory for this run
        run_dir = os.path.join(self.output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Measure training time
        train_start_time = time.time()

        # Train the model
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=(self.verbose >= 2),
        )

        train_end_time = time.time()
        train_time = train_end_time - train_start_time

        # Measure evaluation time
        eval_start_time = time.time()

        # Evaluate the model
        evaluation = model.evaluate(x_test, y_test, verbose=0)

        eval_end_time = time.time()
        eval_time = eval_end_time - eval_start_time

        # Measure prediction time (average per sample)
        pred_times = []
        for _ in range(5):  # Run multiple times for stability
            pred_start_time = time.time()
            model.predict(x_test[:10])  # Predict on a small batch
            pred_end_time = time.time()
            pred_times.append((pred_end_time - pred_start_time) / 10)

        avg_pred_time = sum(pred_times) / len(pred_times)

        # Collect results
        results = {
            "task_name": task_name,
            "model_name": model_name,
            "timestamp": timestamp,
            "performance": dict(zip(model.model.metrics_names, evaluation)),
            "timing": {
                "training_time": train_time,
                "training_time_per_epoch": train_time / epochs,
                "evaluation_time": eval_time,
                "prediction_time_per_sample": avg_pred_time,
            },
            "configuration": {
                "epochs": epochs,
                "batch_size": batch_size,
                "train_samples": len(x_train),
                "test_samples": len(x_test),
                "input_shape": list(x_train.shape[1:]),
                "output_shape": list(y_train.shape[1:]) if y_train.ndim > 1 else [1],
            },
            "history": {
                k: [float(v) for v in history.history[k]] for k in history.history
            },
        }

        # Save results
        self._save_results(results, run_dir)

        # Generate visualizations
        self._generate_visualizations(results, run_dir)

        return results

    def _save_results(self, results: Dict[str, Any], run_dir: str):
        """Save benchmark results to disk."""
        # Save JSON results
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        if self.verbose >= 1:
            print(f"Results saved to {run_dir}/results.json")

    def _generate_visualizations(self, results: Dict[str, Any], run_dir: str):
        """Generate visualizations for benchmark results."""
        # Plot training history
        fig = plot_training_trajectory(
            results["history"],
            metrics=self.metrics,
            title=f"Training History: {results['model_name']} on {results['task_name']}",
        )

        # Save plot
        fig.savefig(os.path.join(run_dir, "training_history.png"), bbox_inches="tight")
        plt.close(fig)

        # Create performance summary plot
        self._plot_performance_summary(results, run_dir)

    def _plot_performance_summary(self, results: Dict[str, Any], run_dir: str):
        """Create a summary plot of benchmark performance."""
        perf = results["performance"]
        timing = results["timing"]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot performance metrics
        metrics = list(perf.keys())
        values = list(perf.values())

        colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
        ax1.bar(metrics, values, color=colors)
        ax1.set_title("Performance Metrics")
        ax1.set_ylabel("Value")
        ax1.set_ylim(0, max(1.0, max(values) * 1.1))

        # Plot timing metrics
        timing_metrics = list(timing.keys())
        timing_values = list(timing.values())

        colors = plt.cm.plasma(np.linspace(0, 1, len(timing_metrics)))
        ax2.bar(timing_metrics, timing_values, color=colors)
        ax2.set_title("Timing Metrics")
        ax2.set_ylabel("Time (seconds)")

        # Adjust layout and save
        plt.tight_layout()
        fig.savefig(
            os.path.join(run_dir, "performance_summary.png"), bbox_inches="tight"
        )
        plt.close(fig)

    def compare_models(
        self,
        models: Dict[str, QuantumModel],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        task_name: str,
        epochs: int = 10,
        batch_size: int = 16,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models on the same benchmark task.

        Args:
            models: Dictionary mapping model names to model instances
            x_train: Training input data
            y_train: Training target data
            x_test: Test input data
            y_test: Test target data
            task_name: Name of the benchmark task
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Dictionary mapping model names to benchmark results
        """
        if self.verbose >= 1:
            print(f"Comparing {len(models)} models on task: {task_name}")

        # Run benchmark for each model
        results = {}
        for model_name, model in models.items():
            results[model_name] = self.run_benchmark(
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                task_name=task_name,
                model_name=model_name,
                epochs=epochs,
                batch_size=batch_size,
            )

        # Generate comparison visualizations
        self._generate_comparison_visualizations(results, task_name)

        return results

    def _generate_comparison_visualizations(
        self, results: Dict[str, Dict[str, Any]], task_name: str
    ):
        """Generate visualizations comparing multiple models."""
        # Create comparison directory
        comparison_dir = os.path.join(
            self.output_dir,
            f"comparison_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(comparison_dir, exist_ok=True)

        # Extract model names
        model_names = list(results.keys())

        # Compare performance metrics
        self._plot_performance_comparison(results, model_names, comparison_dir)

        # Compare timing metrics
        self._plot_timing_comparison(results, model_names, comparison_dir)

        # Save comparison results
        with open(os.path.join(comparison_dir, "comparison_results.json"), "w") as f:
            # Extract and format results for serialization
            comparison_results = {
                model_name: {
                    "performance": results[model_name]["performance"],
                    "timing": results[model_name]["timing"],
                }
                for model_name in model_names
            }
            json.dump(comparison_results, f, indent=2)

    def _plot_performance_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        model_names: List[str],
        output_dir: str,
    ):
        """Plot performance comparison between models."""
        # Extract common metrics across all models
        all_metrics = set()
        for model_name in model_names:
            all_metrics.update(results[model_name]["performance"].keys())

        metrics = sorted(list(all_metrics))

        # Create figure with a subplot for each metric
        fig, axes = plt.subplots(
            len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True
        )

        # Handle case with only one metric
        if len(metrics) == 1:
            axes = [axes]

        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Extract values for each model
            values = []
            labels = []

            for model_name in model_names:
                if metric in results[model_name]["performance"]:
                    values.append(results[model_name]["performance"][metric])
                    labels.append(model_name)

            # Sort by value for better visualization
            sorted_indices = np.argsort(values)
            sorted_values = [values[i] for i in sorted_indices]
            sorted_labels = [labels[i] for i in sorted_indices]

            # Create bar chart
            colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_labels)))
            ax.barh(sorted_labels, sorted_values, color=colors)

            # Add labels
            ax.set_title(f"{metric.capitalize()} Comparison")
            ax.set_xlabel("Value")

            # Add value labels
            for j, v in enumerate(sorted_values):
                ax.text(v + 0.01, j, f"{v:.4f}", va="center")

        # Adjust layout and save
        plt.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "performance_comparison.png"), bbox_inches="tight"
        )
        plt.close(fig)

    def _plot_timing_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        model_names: List[str],
        output_dir: str,
    ):
        """Plot timing comparison between models."""
        # Extract common timing metrics across all models
        all_metrics = set()
        for model_name in model_names:
            all_metrics.update(results[model_name]["timing"].keys())

        timing_metrics = sorted(list(all_metrics))

        # Create figure with a subplot for each metric
        fig, axes = plt.subplots(
            len(timing_metrics), 1, figsize=(10, 4 * len(timing_metrics)), sharex=True
        )

        # Handle case with only one metric
        if len(timing_metrics) == 1:
            axes = [axes]

        # Plot each metric
        for i, metric in enumerate(timing_metrics):
            ax = axes[i]

            # Extract values for each model
            values = []
            labels = []

            for model_name in model_names:
                if metric in results[model_name]["timing"]:
                    values.append(results[model_name]["timing"][metric])
                    labels.append(model_name)

            # Sort by value for better visualization
            sorted_indices = np.argsort(values)
            sorted_values = [values[i] for i in sorted_indices]
            sorted_labels = [labels[i] for i in sorted_indices]

            # Create bar chart
            colors = plt.cm.plasma(np.linspace(0, 1, len(sorted_labels)))
            ax.barh(sorted_labels, sorted_values, color=colors)

            # Add labels
            ax.set_title(f"{metric.replace('_', ' ').capitalize()} Comparison")
            ax.set_xlabel("Time (seconds)")

            # Add value labels
            for j, v in enumerate(sorted_values):
                ax.text(v + 0.01, j, f"{v:.4f}s", va="center")

        # Adjust layout and save
        plt.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "timing_comparison.png"), bbox_inches="tight"
        )
        plt.close(fig)
