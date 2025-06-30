"""
Quantum feature encoding and kernel methods.

This module demonstrates how to leverage quantum circuits for encoding
classical data into quantum states and computing quantum kernels
that may provide advantages over classical kernel methods.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from quantum_nn.circuits import ParameterizedCircuit
from quantum_nn.layers import QuantumLayer


class QuantumFeatureEncoder:
    """
    Encode classical data into quantum feature spaces.

    This class demonstrates how to encode classical data into quantum states,
    potentially accessing feature spaces that are difficult to reach with
    classical methods, due to the exponential dimensionality of quantum Hilbert spaces.

    Quantum encoding can offer computational advantages by:
    1. Efficiently representing high-dimensional data
    2. Accessing non-linear feature maps implicitly
    3. Potentially escaping the curse of dimensionality
    """

    def __init__(
        self,
        n_qubits: int,
        encoding_type: str = "amplitude",
        n_layers: int = 2,
        entanglement: str = "full",
        observables: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize a quantum feature encoder.

        Args:
            n_qubits: Number of qubits in the quantum circuit
            encoding_type: Encoding method ('angle', 'amplitude', or 'basis')
            n_layers: Number of layers in the quantum circuit
            entanglement: Entanglement pattern ('full', 'linear', or 'circular')
            observables: Quantum observables to measure
            random_state: Random seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.observables = observables
        self.random_state = random_state

        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
            tf.random.set_seed(random_state)

        # Initialize preprocessing
        self.scaler = StandardScaler()

        # Build quantum circuit
        self._build_circuit()

    def _build_circuit(self):
        """Build the quantum feature encoding circuit."""
        self.circuit = ParameterizedCircuit(
            n_qubits=self.n_qubits,
            template="strongly_entangling",
            template_kwargs={"pattern": self.entanglement, "n_layers": self.n_layers},
            encoder=self.encoding_type,
            observables=self.observables,
        )

        self.n_params = self.circuit.get_n_params()

        # Initialize random parameters for the circuit
        self.params = np.random.uniform(-np.pi, np.pi, size=self.n_params)

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "QuantumFeatureEncoder":
        """
        Fit the feature encoder to the data.

        Args:
            X: Input data
            y: Target data (unused, included for scikit-learn compatibility)

        Returns:
            Self
        """
        # Scale the input data
        self.scaler.fit(X)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using quantum feature encoding.

        Args:
            X: Input data

        Returns:
            Quantum features
        """
        # Scale the input data
        X_scaled = self.scaler.transform(X)

        # Apply quantum feature encoding
        n_samples = X_scaled.shape[0]
        n_features = self.circuit(self.params).shape[0]

        # Initialize output array
        quantum_features = np.zeros((n_samples, n_features))

        # Process each sample
        for i, x in enumerate(X_scaled):
            # Prepare input for the circuit
            if self.encoding_type == "amplitude":
                # Ensure the input is properly normalized for amplitude encoding
                input_dim = 2**self.n_qubits
                if len(x) > input_dim:
                    # Truncate if input is too large
                    x = x[:input_dim]
                elif len(x) < input_dim:
                    # Pad with zeros if input is too small
                    x = np.pad(x, (0, input_dim - len(x)))

                # Normalize the input vector
                x = x / np.linalg.norm(x)

            # Execute the quantum circuit
            quantum_features[i] = self.circuit(self.params, x)

        return quantum_features

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit the encoder and transform the data.

        Args:
            X: Input data
            y: Target data (unused, included for scikit-learn compatibility)

        Returns:
            Quantum features
        """
        return self.fit(X, y).transform(X)

    def quantum_kernel(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the quantum kernel matrix.

        The quantum kernel K(x,y) measures the similarity between data points
        in the quantum feature space. It's defined as the squared inner product
        of the quantum feature vectors: K(x,y) = |<φ(x)|φ(y)>|².

        This can enable kernel methods to access feature spaces that may be
        inaccessible to classical kernels.

        Args:
            X: First data matrix
            Y: Second data matrix (if None, use X)

        Returns:
            Quantum kernel matrix
        """
        # Transform data to quantum features
        X_quantum = self.transform(X)

        if Y is None:
            Y_quantum = X_quantum
        else:
            Y_quantum = self.transform(Y)

        # Compute kernel matrix
        kernel_matrix = np.zeros((X_quantum.shape[0], Y_quantum.shape[0]))

        for i, x in enumerate(X_quantum):
            for j, y in enumerate(Y_quantum):
                # Compute the squared inner product
                # For normalized quantum states, this is equivalent to the
                # overlap between the corresponding quantum states
                kernel_matrix[i, j] = np.abs(np.dot(x, y)) ** 2

        return kernel_matrix

    def visualize_kernel(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 8),
    ):
        """
        Visualize the quantum kernel matrix.

        Args:
            X: Input data
            y: Class labels for coloring (optional)
            figsize: Figure size
        """
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        # Compute kernel matrix
        kernel_matrix = self.quantum_kernel(X)

        # Create figure
        plt.figure(figsize=figsize)

        # Plot kernel matrix
        im = plt.imshow(kernel_matrix, cmap="viridis")
        plt.colorbar(im, label="Kernel Value")
        plt.title("Quantum Kernel Matrix")
        plt.xlabel("Sample Index")
        plt.ylabel("Sample Index")

        # Add class separators if y is provided
        if y is not None:
            # Find boundaries between classes
            unique_classes = np.unique(y)
            boundaries = []

            for cls in unique_classes[:-1]:
                # Find the index where the class changes
                boundaries.append(np.sum(y == cls))

            # Convert to cumulative indices
            cum_boundaries = np.cumsum(boundaries)

            # Add lines to show class boundaries
            for boundary in cum_boundaries:
                plt.axhline(y=boundary - 0.5, color="red", linestyle="-", alpha=0.7)
                plt.axvline(x=boundary - 0.5, color="red", linestyle="-", alpha=0.7)

        plt.tight_layout()
        return plt.gcf()


class QuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier using quantum kernels for enhanced expressivity.

    This classifier leverages quantum feature encodings to compute
    kernel matrices, potentially allowing access to feature spaces
    that are difficult to reach with classical methods.

    Quantum advantage arises from:
    1. Quantum kernel functions that may be difficult to compute classically
    2. Access to high-dimensional Hilbert spaces without explicit calculation
    3. Natural handling of non-linear feature mappings
    """

    def __init__(
        self,
        n_qubits: int = 4,
        encoding_type: str = "amplitude",
        n_layers: int = 2,
        entanglement: str = "full",
        observables: Optional[List[str]] = None,
        C: float = 1.0,
        kernel_approx: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initialize a quantum kernel classifier.

        Args:
            n_qubits: Number of qubits in the quantum circuit
            encoding_type: Encoding method ('angle', 'amplitude', or 'basis')
            n_layers: Number of layers in the quantum circuit
            entanglement: Entanglement pattern ('full', 'linear', or 'circular')
            observables: Quantum observables to measure
            C: Regularization parameter
            kernel_approx: Whether to use kernel approximation for larger datasets
            random_state: Random seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.observables = observables
        self.C = C
        self.kernel_approx = kernel_approx
        self.random_state = random_state

        # Initialize quantum feature encoder
        self.encoder = QuantumFeatureEncoder(
            n_qubits=n_qubits,
            encoding_type=encoding_type,
            n_layers=n_layers,
            entanglement=entanglement,
            observables=observables,
            random_state=random_state,
        )

        # Initialize classifier
        from sklearn.svm import SVC

        self.svc = SVC(
            kernel="precomputed", C=self.C, probability=True, random_state=random_state
        )

        # For kernel approximation
        self.use_approx = False
        self.approx_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelClassifier":
        """
        Fit the quantum kernel classifier.

        Args:
            X: Training data
            y: Target labels

        Returns:
            Self
        """
        # Fit the encoder
        self.encoder.fit(X)

        # Check if we should use kernel approximation
        if self.kernel_approx and X.shape[0] > 1000:
            self.use_approx = True
            # Use quantum features directly
            self.approx_features = self.encoder.transform(X)
            from sklearn.svm import SVC

            # Switch to RBF kernel on quantum features
            self.svc = SVC(
                kernel="rbf", C=self.C, probability=True, random_state=self.random_state
            )
            self.svc.fit(self.approx_features, y)
        else:
            # Compute the quantum kernel matrix
            K = self.encoder.quantum_kernel(X)
            # Fit the SVM with the precomputed kernel
            self.svc.fit(K, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Test data

        Returns:
            Predicted class labels
        """
        if self.use_approx:
            # Transform to quantum features
            X_quantum = self.encoder.transform(X)
            # Predict using the SVM
            return self.svc.predict(X_quantum)
        else:
            # Compute the kernel between test and training data
            K_test = self.encoder.quantum_kernel(X, self.svc._fit_X)
            # Predict using the SVM
            return self.svc.predict(K_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Test data

        Returns:
            Class probabilities
        """
        if self.use_approx:
            # Transform to quantum features
            X_quantum = self.encoder.transform(X)
            # Predict probabilities using the SVM
            return self.svc.predict_proba(X_quantum)
        else:
            # Compute the kernel between test and training data
            K_test = self.encoder.quantum_kernel(X, self.svc._fit_X)
            # Predict probabilities using the SVM
            return self.svc.predict_proba(K_test)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the accuracy score.

        Args:
            X: Test data
            y: True labels

        Returns:
            Accuracy score
        """
        return accuracy_score(y, self.predict(X))

    def evaluation_report(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.

        Args:
            X: Test data
            y: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            roc_auc_score,
            roc_curve,
        )

        # Make predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        # Number of classes
        n_classes = len(np.unique(y))

        # Basic metrics
        report_dict = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

        # ROC AUC for binary classification
        if n_classes == 2:
            report_dict["roc_auc"] = roc_auc_score(y, y_proba[:, 1])
            fpr, tpr, thresholds = roc_curve(y, y_proba[:, 1])
            report_dict["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }

        # Add classification report
        clf_report = classification_report(y, y_pred, output_dict=True)
        report_dict["classification_report"] = clf_report

        return report_dict

    def visualize_decision_boundary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: List[int] = [0, 1],
        resolution: int = 100,
        figsize: Tuple[int, int] = (10, 8),
    ):
        """
        Visualize the decision boundary for 2D data.

        Args:
            X: Input data
            y: Class labels
            feature_indices: Indices of features to plot
            resolution: Resolution of the grid
            figsize: Figure size
        """
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        # Extract the two features
        X_2d = X[:, feature_indices]

        # Create a mesh grid
        x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
        y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )

        # Create full input data for prediction
        if X.shape[1] > 2:
            # For higher dimensional data, use the mean of other features
            mean_features = np.mean(X, axis=0)
            mesh_input = np.zeros((resolution * resolution, X.shape[1]))

            # Set all features to the mean value
            for i in range(X.shape[1]):
                mesh_input[:, i] = mean_features[i]

            # Override the two features we're visualizing
            mesh_input[:, feature_indices[0]] = xx.ravel()
            mesh_input[:, feature_indices[1]] = yy.ravel()
        else:
            # For 2D data, we can just use the mesh directly
            mesh_input = np.c_[xx.ravel(), yy.ravel()]

        # Predict on the mesh
        Z = self.predict(mesh_input)
        Z = Z.reshape(xx.shape)

        # Create figure
        plt.figure(figsize=figsize)

        # Plot decision boundary
        cmap_viridis = plt.get_cmap('viridis')
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_viridis)

        # Plot data points
        scatter = plt.scatter(
            X_2d[:, 0], X_2d[:, 1], c=y, edgecolors="k", alpha=0.9, cmap=cmap_viridis
        )

        plt.colorbar(scatter, label="Class")
        plt.title("Quantum Kernel Decision Boundary")
        plt.xlabel(f"Feature {feature_indices[0]}")
        plt.ylabel(f"Feature {feature_indices[1]}")

        plt.tight_layout()
        return plt.gcf()
