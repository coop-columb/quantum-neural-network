# Changelog

All notable changes to this project will be documented in this file.

The format adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0/).

---

## [Unreleased]
### Added
- Comprehensive docstrings and interface clarifications for all public classes and functions.
- Improved parameter, return type, and error documentation for major modules.
- Project-wide code formatting using Black for consistency and CI compliance.

### Changed
- Refactored some function signatures and argument names for clarity (no business logic changes).
- Updated code style to meet Black and linting standards throughout `src` and `tests`.

### Fixed
- Resolved minor inconsistencies in module imports and interface definitions.
- Fixed various formatting issues identified by Black in the CI workflow.

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
