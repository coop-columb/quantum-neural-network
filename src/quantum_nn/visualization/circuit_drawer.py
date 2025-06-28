"""
Circuit visualization tools.

This module provides functions to visualize quantum circuits
using various representation formats.
"""

from typing import Dict, List, Optional, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from quantum_nn.circuits import ParameterizedCircuit


def draw_circuit(
    circuit: Union[ParameterizedCircuit, qml.QNode],
    params: Optional[np.ndarray] = None,
    inputs: Optional[np.ndarray] = None,
    show_params: bool = True,
    figsize: tuple = (10, 6),
    style: str = "default",
    output_format: str = "matplotlib",
    filename: Optional[str] = None,
) -> Any:
    """
    Draw a quantum circuit diagram.

    Args:
        circuit: Quantum circuit to visualize
        params: Circuit parameters (if ParameterizedCircuit is provided)
        inputs: Input data (optional)
        show_params: Whether to show parameter values
        figsize: Figure size for matplotlib output
        style: Visualization style ('default', 'blueprint', 'sketch')
        output_format: Output format ('matplotlib', 'text', 'latex')
        filename: File to save the visualization to (optional)

    Returns:
        Visualization object (matplotlib figure or string)
    """
    # Extract QNode from ParameterizedCircuit if necessary
    qnode = circuit.circuit if isinstance(circuit, ParameterizedCircuit) else circuit

    # Set random parameters if not provided and circuit requires them
    if params is None and isinstance(circuit, ParameterizedCircuit):
        params = np.random.uniform(-np.pi, np.pi, size=circuit.get_n_params())

    # Create drawing function
    def circuit_function():
        if isinstance(circuit, ParameterizedCircuit):
            # Execute the circuit function with our parameters and inputs
            circuit._circuit_fn(params, inputs)
        else:
            # For a raw QNode, just execute it
            args = []
            if params is not None:
                args.append(params)
            if inputs is not None:
                args.append(inputs)
            qnode.func(*args)

    # Generate visualization based on output format
    if output_format.lower() == "matplotlib":
        # Use PennyLane's built-in drawer
        fig, ax = plt.subplots(figsize=figsize)
        drawer = qml.draw_mpl(circuit_function)
        drawer(ax=ax)

        # Apply style
        if style == "blueprint":
            ax.set_facecolor("#f0f8ff")
            plt.setp(ax.spines.values(), color="navy")
            ax.tick_params(colors="navy")
            ax.set_title("Quantum Circuit Blueprint", color="navy")
        elif style == "sketch":
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = "Comic Sans MS"
            ax.set_title("Quantum Circuit Sketch")
        else:
            ax.set_title("Quantum Circuit Diagram")

        if filename:
            plt.savefig(filename, bbox_inches="tight")

        return fig

    elif output_format.lower() == "text":
        # Use PennyLane's ASCII drawer
        text_diagram = qml.draw(circuit_function)

        if filename:
            with open(filename, "w") as f:
                f.write(text_diagram)

        return text_diagram

    elif output_format.lower() == "latex":
        # Use PennyLane's LaTeX drawer
        latex_diagram = qml.draw_mpl(circuit_function, style="sketch")

        if filename and filename.endswith(".tex"):
            tikz_code = qml.draw(
                circuit_function,
                expansion_strategy="device",
                output_format="latex_source",
            )
            with open(filename, "w") as f:
                f.write(tikz_code)

        return latex_diagram

    else:
        raise ValueError(f"Unknown output format: {output_format}")
