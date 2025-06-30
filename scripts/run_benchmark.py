#!/usr/bin/env python
"""
Script for running quantum neural network benchmarks.

This script demonstrates how to benchmark quantum neural networks
against classical models on standard tasks.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from quantum_nn.benchmarks import (
    BenchmarkRunner,
    get_benchmark_task,
    run_classical_comparison,
)
from quantum_nn.circuits import ParameterizedCircuit
from quantum_nn.models import QuantumModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run quantum neural network benchmarks"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="binary_classification",
        choices=[
            "binary_classification",
            "multiclass_classification",
            "regression",
            "moons",
            "digits",
            "quantum_simulation",
            "entanglement_classification",
        ],
        help="Benchmark task",
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--quantum-layers", type=int, default=2, help="Number of quantum layers"
    )
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare with classical models"
    )
    parser.add_argument(
        "--output", type=str, default="./experiments/results", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def create_quantum_model(task_info, n_qubits, n_layers):
    """Create a quantum model based on task information."""
    # Determine input and output dimensions
    input_dim = task_info["n_features"]

    if task_info["task_type"] == "classification":
        output_dim = task_info["n_classes"]
        if output_dim == 2:
            output_dim = 1  # Binary classification
            output_activation = "sigmoid"
            loss = "binary_crossentropy"
        else:
            output_activation = "softmax"
            loss = "categorical_crossentropy"
    else:  # regression
        output_dim = task_info.get("n_outputs", 1)
        output_activation = "linear"
        loss = "mse"

    # Create circuit with appropriate number of qubits
    circuit = ParameterizedCircuit(
        n_qubits=n_qubits,
        n_layers=n_layers,
        template="strongly_entangling",
        template_kwargs={"pattern": "full"},
    )

    # Create model
    model = QuantumModel(
        [
            {"type": "quantum", "n_qubits": n_qubits, "n_layers": n_layers},
            {"type": "dense", "units": 16, "activation": "relu"},
            {"type": "dense", "units": output_dim, "activation": output_activation},
        ]
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=loss,
        metrics=["accuracy"] if task_info["task_type"] == "classification" else ["mae"],
    )

    return model


def main():
    """Run benchmark."""
    # Parse command line arguments
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Get benchmark task
    print(f"Loading benchmark task: {args.task}")
    x_train, y_train, x_test, y_test, task_info = get_benchmark_task(
        task_name=args.task,
        n_samples=args.samples,
        random_state=args.seed,
        task_kwargs={"n_qubits": args.qubits},
    )

    # Create quantum model
    print(
        f"Creating quantum model with {args.qubits} qubits and {args.quantum_layers} layers"
    )
    model = create_quantum_model(task_info, args.qubits, args.quantum_layers)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"benchmark_{args.task}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save task information
    with open(os.path.join(output_dir, "task_info.json"), "w") as f:
        # Convert non-serializable objects to strings
        serializable_info = task_info.copy()
        for key, value in task_info.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                serializable_info[key] = str(value)

        json.dump(serializable_info, f, indent=2)

    # Run benchmark
    if args.compare:
        print("Running comparison with classical models")
        results = run_classical_comparison(
            quantum_model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            task_name=args.task,
            quantum_model_name=f"quantum_{args.qubits}qubits_{args.quantum_layers}layers",
            classical_models=["mlp"],  # Add more models if needed
            task_type=task_info["task_type"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=output_dir,
        )
    else:
        print("Running quantum model benchmark")
        runner = BenchmarkRunner(output_dir=output_dir)
        results = runner.run_benchmark(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            task_name=args.task,
            model_name=f"quantum_{args.qubits}qubits_{args.quantum_layers}layers",
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    print(f"Benchmark completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
