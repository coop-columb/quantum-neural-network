# Changelog

All notable changes to this project will be documented in this file.

The format adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]
### Added
- Initial, modular implementation of quantum neural network (QNN) framework.
- Core quantum circuit templates (e.g., strongly entangling, quantum convolution, quantum residual).
- Data encoding schemes: amplitude, angle, basis, and hybrid encoding.
- Quantum-aware optimizers: parameter-shift rule, quantum natural gradient, SPSA.
- Integration with both Qiskit and PennyLane backends.
- TensorFlow-compatible quantum layers for hybrid models.
- Example Jupyter notebooks for classification and regression with QNNs.
- Benchmark runner for classical vs. quantum model performance.
- Unit and integration test suite covering major modules.

### Changed
- N/A

### Fixed
- N/A

---

## [0.1.0] - 2025-07-01
### Added
- Project scaffolding, repository structure, and documentation.
- Initial implementation of `quantum_nn` core modules:
  - `circuits/`: Parameterized and template quantum circuits.
  - `layers/`: Quantum TensorFlow layer abstraction.
  - `models/`: Sequential quantum model assembly.
  - `optimizers/`: Quantum-aware optimizer interfaces.
- Example scripts and benchmarks.
- CI configuration and code style enforcement.

---

[Unreleased]: https://github.com/coop-columb/quantum-neural-network/compare/main...HEAD
[0.1.0]: https://github.com/coop-columb/quantum-neural-network/releases/tag/0.1.0
