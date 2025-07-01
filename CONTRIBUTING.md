# Contributing to Quantum Neural Network Framework

Thank you for your interest in contributing to the Quantum Neural Network Framework!  
This document describes the contribution process, coding standards, and review protocols to ensure a high-quality, consistent, and innovative codebase for quantum machine learning research.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Requesting Features](#requesting-features)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Development Guidelines](#development-guidelines)
  - [Project Structure](#project-structure)
  - [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)
- [Contact](#contact)

---

## Code of Conduct

All contributors are expected to adhere to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).  
Be respectful, inclusive, and constructive.

---

## How to Contribute

### Reporting Bugs

1. **Search existing issues** to ensure the bug hasn’t already been reported.
2. [Open a new issue](https://github.com/coop-columb/quantum-neural-network/issues/new) using the "Bug report" template.
3. Provide:
   - A clear, descriptive title.
   - Steps to reproduce (with code snippets or data).
   - Expected and actual behavior.
   - Environment details: OS, Python version, backend (e.g., Qiskit, PennyLane), and framework versions.

### Requesting Features

1. Search existing issues to avoid duplicates.
2. Open a feature request issue using the "Feature request" template.
3. Clearly describe:
   - The problem or opportunity.
   - Proposed solution or API change.
   - Potential use cases and impact.

### Submitting Pull Requests

1. **Fork the repository** and create your own feature branch.
2. Write concise, well-structured, and well-documented code.
3. Ensure all tests pass (`pytest`).
4. Follow code style checks (`black`, `isort`, `mypy`).
5. Add or update documentation and examples.
6. Reference related issues or discussions in your PR description.
7. Submit a pull request against the `main` branch, following the PR template.

**All significant changes (API, internals, or architecture) should be discussed in an issue before opening a PR.**

---

## Development Guidelines

### Project Structure

- `src/quantum_nn/circuits/`: Quantum circuit templates and parameterized circuits
- `src/quantum_nn/layers/`: Quantum-to-classical neural network layers (TensorFlow integration)
- `src/quantum_nn/models/`: Model architectures (sequential, hybrid, etc.)
- `src/quantum_nn/optimizers/`: Quantum-aware optimizers (parameter-shift, SPSA, etc.)
- `docs/`: Technical and user documentation
- `notebooks/`: Jupyter notebooks with practical examples
- `tests/`: Unit and integration tests

### Coding Standards

- **Python 3.10+**  
- Follow [PEP 8](https://pep8.org/) and [PEP 484](https://peps.python.org/pep-0484/) (type hints).
- Use [Black](https://black.readthedocs.io/en/stable/) for formatting and [isort](https://pycqa.github.io/isort/) for imports.
- Include docstrings for all public modules, functions, classes, and methods (Google or NumPy style).
- Use meaningful variable and function names.
- Write atomic, testable, and reusable code.
- Prefer explicit over implicit behavior.

---

## Testing

- All new features and bug fixes must be accompanied by appropriate tests in `tests/`.
- Use `pytest` for running tests and `pytest-cov` for coverage.
- Achieve and maintain high code coverage; target >90%.
- Test classical and quantum backends (Qiskit, PennyLane).
- Run tests locally before submitting.

---

## Documentation

- Update user and developer documentation in `docs/` as needed.
- Public APIs must be documented with usage examples.
- For new features, add or update example notebooks in `notebooks/`.
- Use [Sphinx](https://www.sphinx-doc.org/) for generating documentation.

---

## Community

- Join discussions in [GitHub Discussions](https://github.com/coop-columb/quantum-neural-network/discussions).
- Respond thoughtfully to issues and PR feedback.
- Propose RFCs for major changes (see `docs/ARCHITECTURE.md` and `docs/DESIGN.md`).

---

## Contact

Questions or suggestions?  
Open an [issue](https://github.com/coop-columb/quantum-neural-network/issues), start a [discussion](https://github.com/coop-columb/quantum-neural-network/discussions), or email the maintainer: your.email@example.com

---

Thank you for helping to build the future of quantum machine learning!
