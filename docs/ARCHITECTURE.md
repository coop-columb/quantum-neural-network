# Quantum Neural Network Framework – Architecture

This document details the architecture of the Quantum Neural Network (QNN) Framework, describing the core modules, data flow, extensibility points, and integration between quantum and classical machine learning components.

---

## Overview

The QNN framework is designed for hybrid quantum-classical machine learning, enabling flexible model construction, simulation, and benchmarking of quantum neural networks. The architecture supports multiple quantum backends (Qiskit, PennyLane), integrates with TensorFlow for classical machine learning, and provides components for circuit design, data encoding, optimization, and benchmarking.

---

## High-Level Architecture

```
+----------------+
|   Classical    |
|   Data Source  |
+-------+--------+
        |
        v
+-------------------+      +--------------------------+
| Data Encoding     |----->| Quantum Circuit          |
| (Amplitude, etc.) |      | (Templates, Parameterized)|
+-------------------+      +--------------------------+
        |                             |
        v                             v
+-------------------+      +--------------------------+
| Quantum Layer     |      | Classical Layers         |
| (QNN, Hybrid)     |----->| (Dense, Activation, etc.)|
+-------------------+      +--------------------------+
        |
        v
+-------------------+
| Quantum Model     |
| (Sequential/Hybrid)|
+-------------------+
        |
        v
+-------------------+
| Optimizer         |
| (Parameter-Shift, |
| Quantum Nat. Grad)|
+-------------------+
        |
        v
+-------------------+
| Training & Eval   |
+-------------------+
        |
        v
+-------------------+
| Benchmark Suite   |
+-------------------+
```

---

## Main Components

### 1. `circuits/`

- **Purpose:** Provides parameterized quantum circuit templates (e.g., strongly entangling layers, quantum convolution, quantum residual).
- **Backend Support:** Qiskit and PennyLane (via abstract interfaces).
- **Key Classes:**  
  - `ParameterizedCircuit` (base class)
  - Backend-specific implementations

### 2. `layers/`

- **Purpose:** Implements `QuantumLayer`, a TensorFlow-compatible layer that integrates quantum circuits into classical models.
- **Features:**  
  - Batch processing for quantum circuits  
  - Automatic differentiation support  
  - Measurement and observable selection

### 3. `models/`

- **Purpose:** Model assembly for quantum/hybrid neural networks.
- **Structure:**  
  - Sequential model pattern  
  - Arbitrary stacking of quantum and classical layers  
  - Supports Keras Model API for training, evaluation, and export

### 4. `optimizers/`

- **Purpose:** Quantum-aware optimization algorithms.
- **Implemented:**  
  - Parameter-shift rule  
  - Quantum natural gradient  
  - Simultaneous Perturbation Stochastic Approximation (SPSA)
- **Extensibility:** Easily add new optimizers by subclassing the optimizer base class.

### 5. `data_encoding/`

- **Purpose:** Schemes for encoding classical data into quantum states.
- **Included:**  
  - Amplitude encoding  
  - Angle encoding  
  - Basis encoding  
  - Hybrid encoding

### 6. `benchmarks/` & `notebooks/`

- **Purpose:** Benchmark runner for classical vs. quantum models; Jupyter notebooks for demos.
- **Features:**  
  - Configurable experiment scripts  
  - Metrics logging and visualization

### 7. `docs/`

- **Purpose:** Technical documentation, design explanations, and API references.

---

## Data Flow

1. **Data ingestion:** Classical data is ingested from user code or dataset loaders.
2. **Encoding:** Data is encoded into quantum states using selected encoding schemes.
3. **Quantum processing:** Encoded data passes through parameterized quantum circuits.
4. **Measurement:** Quantum layer outputs are measured and mapped to classical representations.
5. **Integration:** Quantum outputs are integrated with classical neural network layers.
6. **Optimization:** Training uses quantum-aware optimizers for parameter updates.
7. **Benchmarking:** Model performance is compared against classical baselines.

---

## Extensibility

- **Backends:** Plug-in architecture for new quantum simulators/hardware.
- **Circuits:** Add new circuit templates in `circuits/`.
- **Layers:** Compose new hybrid layers by subclassing `QuantumLayer`.
- **Optimizers:** Implement novel quantum/classical optimizers.
- **Benchmarks:** Add new benchmarks via the benchmarking script interface.
- **Documentation:** All core and contributed modules must be documented in `docs/`.

---

## Code Quality & Testing

- **Type annotations:** Mandatory for all public APIs.
- **CI checks:** Includes formatting, linting, static analysis, and automated tests.
- **Coverage:** >90% coverage required for all modules.
- **Notebook validation:** Example notebooks are tested for reproducibility.

---

## Dependencies

- **Quantum:** Qiskit, PennyLane
- **Classical ML:** TensorFlow, NumPy, SciPy, Pandas
- **Documentation:** Sphinx
- **Visualization:** Matplotlib

---

## References

- See `/docs/DESIGN.md` for API/class diagrams and deeper technical detail.
- See `/docs/ROADMAP.md` for planned features and architectural evolution.

---
