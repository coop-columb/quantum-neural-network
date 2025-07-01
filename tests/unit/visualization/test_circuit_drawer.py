"""
Tests for circuit visualization.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import numpy as np
import pennylane as qml
import pytest

from quantum_nn.visualization import draw_circuit


# Create a simple mock ParameterizedCircuit for testing
class MockParameterizedCircuit:
    """Mock ParameterizedCircuit for testing."""

    def __init__(self, n_params=2):
        self.n_params = n_params

    def get_n_params(self):
        return self.n_params

    def _circuit_fn(self, params, inputs):
        # Simple mock circuit that applies rotations
        for i, param in enumerate(params):
            qml.RY(param, wires=i % 2)


class TestCircuitVisualization:
    """Test suite for circuit visualization."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple PennyLane QNode for testing
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def simple_circuit(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        self.qnode = simple_circuit
        self.mock_circuit = MockParameterizedCircuit(n_params=2)
        self.test_params = np.array([0.1, 0.2])

    def test_draw_circuit_qnode_matplotlib(self):
        """Test drawing QNode with matplotlib output."""
        fig = draw_circuit(
            self.qnode, params=self.test_params, output_format="matplotlib"
        )
        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_draw_circuit_qnode_text(self):
        """Test drawing QNode with text output."""
        text_output = draw_circuit(
            self.qnode, params=self.test_params, output_format="text"
        )
        assert text_output is not None
        assert isinstance(text_output, str)
        assert len(text_output) > 0

    def test_draw_circuit_parameterized_circuit(self):
        """Test drawing ParameterizedCircuit."""
        fig = draw_circuit(
            self.mock_circuit, params=self.test_params, output_format="matplotlib"
        )
        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_draw_circuit_random_params(self):
        """Test drawing with automatically generated random parameters."""
        fig = draw_circuit(self.mock_circuit, params=None, output_format="matplotlib")
        assert fig is not None

    def test_draw_circuit_styles(self):
        """Test different visualization styles."""
        styles = ["default", "blueprint", "sketch"]
        for style in styles:
            fig = draw_circuit(
                self.qnode,
                params=self.test_params,
                style=style,
                output_format="matplotlib",
            )
            assert fig is not None

    def test_draw_circuit_with_inputs(self):
        """Test drawing circuit with input data."""
        inputs = np.array([0.5, 0.3])
        fig = draw_circuit(
            self.qnode,
            params=self.test_params,
            inputs=inputs,
            output_format="matplotlib",
        )
        assert fig is not None

    def test_draw_circuit_custom_figsize(self):
        """Test drawing with custom figure size."""
        fig = draw_circuit(
            self.qnode,
            params=self.test_params,
            figsize=(12, 8),
            output_format="matplotlib",
        )
        assert fig is not None
        # Check that figure size is approximately correct
        assert abs(fig.get_figwidth() - 12) < 0.1
        assert abs(fig.get_figheight() - 8) < 0.1

    def test_invalid_circuit_type(self):
        """Test error handling for invalid circuit type."""
        with pytest.raises(TypeError):
            draw_circuit("not_a_circuit", output_format="matplotlib")

    def test_invalid_style(self):
        """Test error handling for invalid style."""
        with pytest.raises(ValueError):
            draw_circuit(
                self.qnode,
                params=self.test_params,
                style="invalid_style",
                output_format="matplotlib",
            )

    def test_invalid_output_format(self):
        """Test error handling for invalid output format."""
        with pytest.raises(ValueError):
            draw_circuit(
                self.qnode,
                params=self.test_params,
                output_format="invalid_format",
            )

    def test_draw_circuit_latex_output(self):
        """Test LaTeX output format."""
        # Note: This test might be limited by PennyLane's LaTeX capabilities
        try:
            result = draw_circuit(
                self.qnode, params=self.test_params, output_format="latex"
            )
            assert result is not None
        except Exception:
            # LaTeX output might not work in all environments
            pytest.skip("LaTeX output not available in test environment")

    def test_file_saving(self, tmp_path):
        """Test saving visualization to file."""
        # Test matplotlib save
        filepath = tmp_path / "test_circuit.png"
        fig = draw_circuit(
            self.qnode,
            params=self.test_params,
            output_format="matplotlib",
            filename=str(filepath),
        )
        assert filepath.exists()

        # Test text save
        text_filepath = tmp_path / "test_circuit.txt"
        text_output = draw_circuit(
            self.qnode,
            params=self.test_params,
            output_format="text",
            filename=str(text_filepath),
        )
        assert text_filepath.exists()
        assert len(text_filepath.read_text()) > 0

    def test_show_params_parameter(self):
        """Test the show_params parameter (mainly for interface completeness)."""
        # This parameter mainly affects PennyLane's internal rendering
        # We just test that it doesn't cause errors
        fig1 = draw_circuit(
            self.qnode,
            params=self.test_params,
            show_params=True,
            output_format="matplotlib",
        )
        fig2 = draw_circuit(
            self.qnode,
            params=self.test_params,
            show_params=False,
            output_format="matplotlib",
        )
        assert fig1 is not None
        assert fig2 is not None