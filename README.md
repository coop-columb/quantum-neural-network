# Quantum Neural Network Framework

A comprehensive implementation of quantum neural networks integrating quantum computing principles with classical machine learning techniques.

## Overview

This project implements a flexible framework for quantum neural networks that can be used for various machine learning tasks. The framework provides:

- Quantum circuit implementations with various templates
- Data encoding schemes for classical-to-quantum conversion
- Neural network integration layers compatible with TensorFlow
- Quantum-aware optimization techniques
- Benchmark comparison suite against classical methods
- Visualization tools for quantum states and training dynamics

## Installation

```bash
# Clone the repository
git clone https://github.com/coop-columb/quantum-neural-network.git
cd quantum-neural-network

# Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e .
```

## Requirements

- Python 3.8+
- TensorFlow 2.4+
- TensorFlow Quantum 0.5+
- Cirq 0.13+
- Qiskit 0.32+

See `pyproject.toml` for a complete list of dependencies.

## Usage

```python
from quantum_nn.models import QuantumModel
from quantum_nn.circuits import ParameterizedCircuit
from quantum_nn.layers import QuantumLayer
from quantum_nn.optimizers import ParameterShiftOptimizer

# Create a quantum circuit
circuit = ParameterizedCircuit(n_qubits=4)
circuit.add_layer_template('strongly_entangling')

# Create a quantum layer
q_layer = QuantumLayer(circuit, measurement_indices=[0, 1])

# Create a quantum model
model = QuantumModel([q_layer])

# Compile with quantum-aware optimizer
optimizer = ParameterShiftOptimizer()
model.compile(optimizer=optimizer, loss='mse')

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)
```

## Documentation

For detailed documentation, see the `/docs` directory or visit our [documentation website](https://coop-columb.github.io/quantum-neural-network).

## Examples

Explore the `/notebooks` directory for interactive Jupyter notebook examples demonstrating various use cases.

## Benchmarks

Run benchmark comparisons using:

```bash
python scripts/run_benchmark.py --task classification --qubits 4 --classical-model mlp
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.