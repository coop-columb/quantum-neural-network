# Quantum Neural Network Framework – Design Document

This document provides a detailed technical design for the Quantum Neural Network (QNN) Framework, elaborating on the architecture, major interfaces, key algorithms, extensibility mechanisms, and design rationale.

---

## 1. Design Goals

- **Modularity:** Decouple quantum circuit, layer, model, optimizer, and backend components for independent development and extension.
- **Hybridization:** Enable seamless integration of quantum and classical neural network layers (supporting hybrid models).
- **Backend Agnosticism:** Abstract quantum circuit execution to support both Qiskit and PennyLane, with minimal user-side changes.
- **Extensibility:** Facilitate addition of new circuit templates, encoding schemes, optimizers, or backends.
- **Reproducibility:** Ensure deterministic results where possible for scientific research.
- **Performance:** Efficient batch processing, support for GPU acceleration in classical components, and fast quantum simulation.

---

## 2. Core Modules and Interfaces

### 2.1. Circuits (`src/quantum_nn/circuits/`)

- **Base Class:** `ParameterizedCircuit`
  - Abstract interface for all quantum circuits.
  - Defines methods for adding layers (e.g., `add_layer_template(name, **kwargs)`), binding parameters, and exporting to backend-specific formats.

- **Backend Implementations:**
  - `QiskitCircuit(ParameterizedCircuit)`
  - `PennyLaneCircuit(ParameterizedCircuit)`
  - Each implements methods for building, parameter binding, and measurement extraction.

- **Templates:**
  - Strongly entangling layers, quantum convolution, quantum residual, and more.
  - Example: `add_layer_template('strongly_entangling', depth=3)`

### 2.2. Data Encoding (`src/quantum_nn/data_encoding/`)

- **Strategies:**
  - Amplitude encoding: `AmplitudeEncoder`
  - Angle encoding: `AngleEncoder`
  - Basis encoding: `BasisEncoder`
  - Hybrid schemes

- **Interface:**
  - All encoders inherit from `BaseEncoder`, with a `.encode(data)` method returning quantum-ready states or gate sequences.

### 2.3. Quantum Layers (`src/quantum_nn/layers/`)

- **QuantumLayer:**  
  - Inherits from `tf.keras.layers.Layer`.
  - Accepts a `ParameterizedCircuit` and measurement indices.
  - Handles batch processing and vectorization for quantum circuit execution.
  - Integrates with TensorFlow’s automatic differentiation for hybrid training.
  - Supports multiple measurement strategies (expectation, sample, probability).

### 2.4. Models (`src/quantum_nn/models/`)

- **QuantumModel:**  
  - Subclasses Keras `Model` for quantum+classical layer compositions.
  - Methods for compilation, training, evaluation, and serialization.
  - Accepts any combination of quantum and classical layers.

### 2.5. Optimizers (`src/quantum_nn/optimizers/`)

- **Implemented:**
  - `ParameterShiftOptimizer`: Uses parameter-shift rule for gradient estimation.
  - `QuantumNaturalGradientOptimizer`: Natural gradient adaptation for quantum circuits.
  - `SPSAOptimizer`: Simultaneous Perturbation Stochastic Approximation for noisy optimization.

- **Extensible:**  
  - All optimizers derive from a base interface with `step`, `apply_gradients`, and `get_config` methods.

### 2.6. Benchmarks & Utilities

- **Benchmark Runner:**  
  - Scriptable entry point for running experiments comparing classical and quantum models.
  - Configurable via CLI arguments and YAML files.

- **Visualization Tools:**  
  - Circuit diagrams (via Qiskit/PennyLane).
  - Loss landscapes, expectation value trajectories, and state vector visualizations.

---

## 3. Key Algorithms

### 3.1. Parameter-Shift Rule

- Used for exact gradients of quantum circuits with respect to parameters.
- Implemented via finite differencing at shifted parameter values, for all differentiable gates.

### 3.2. Quantum Natural Gradient

- Adapts classical natural gradient to the quantum case using the Fubini-Study metric.
- Approximates quantum Fisher information matrix for robust, geometry-aware updates.

### 3.3. SPSA

- Supports optimization in the presence of hardware noise and shot noise.
- Reduces the number of circuit evaluations per gradient step.

---

## 4. API & Class Diagrams

### 4.1. Core Class Diagram (UML)

```
+-------------------------+
| ParameterizedCircuit    |<----------+
+-------------------------+           |
| +add_layer_template()   |           |
| +bind_parameters()      |           |
+-------------------------+           |
        ^                             |
        |                             |
+----------------------+   +-----------------------+
| QiskitCircuit        |   | PennyLaneCircuit      |
+----------------------+   +-----------------------+
        ^                             ^
        |                             |
+-------------------------------------+
|           QuantumLayer              |
+-------------------------------------+
        ^
        |
+---------------------+
|    QuantumModel     |
+---------------------+
```

### 4.2. Example Usage

```python
# Create circuit and quantum layer
circuit = ParameterizedCircuit(n_qubits=4)
circuit.add_layer_template('strongly_entangling', depth=2)
q_layer = QuantumLayer(circuit, measurement_indices=[0])

# Hybrid model with classical and quantum layers
model = QuantumModel([
    tf.keras.layers.Dense(8, activation='relu'),
    q_layer,
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=ParameterShiftOptimizer(), loss='mse')
```

---

## 5. Design Rationale

- **Abstract interfaces** allow use of multiple backends (Qiskit, PennyLane) and rapid prototyping.
- **Batching and vectorization** are prioritized for efficient quantum simulation and training.
- **Keras compatibility** enables use of callbacks, metrics, and serialization.
- **Extensible registry** for circuits, encoders, and optimizers simplifies research and experimentation.

---

## 6. Limitations and Future Improvements

- Real quantum hardware support is limited by current backend APIs.
- Circuit depth and qubit count are constrained in simulators for practical runtimes.
- Planned: support for additional hardware (IonQ, Braket), more expressive circuit templates, distributed training, and advanced visualization.

---

See also [ARCHITECTURE.md](./ARCHITECTURE.md) and [ROADMAP.md](./ROADMAP.md) for further context.
