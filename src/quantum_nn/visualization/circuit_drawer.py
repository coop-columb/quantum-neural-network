"""
Circuit visualization tools.

This module provides functions to visualize quantum circuits
using various representation formats.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from quantum_nn.circuits import ParameterizedCircuit


def draw_circuit(
    circuit: Union[ParameterizedCircuit, qml.QNode],
    params: Optional[np.ndarray] = None,
    inputs: Optional[np.ndarray] = None,
    show_params: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    style: str = "default",
    output_format: str = "matplotlib",
    filename: Optional[str] = None,
) -> Any:
    """
    Draw a quantum circuit diagram using various visualization formats.

    This function provides flexible visualization of quantum circuits with support
    for different output formats, styling options, and parameter display.

    Args:
        circuit: Quantum circuit to visualize. Can be either a ParameterizedCircuit
            from the quantum_nn framework or a PennyLane QNode.
        params: Circuit parameters for ParameterizedCircuit. If None and circuit
            requires parameters, random values in [-π, π] will be generated.
        inputs: Input data for the circuit (optional). Used when circuit accepts
            classical input data.
        show_params: Whether to display parameter values in the visualization.
            Only affects certain output formats.
        figsize: Figure size as (width, height) for matplotlib output format.
        style: Visualization style. Options:
            - 'default': Standard circuit diagram
            - 'blueprint': Blue-tinted technical drawing style
            - 'sketch': Hand-drawn appearance with serif font
        output_format: Output format for the visualization. Options:
            - 'matplotlib': Interactive matplotlib figure
            - 'text': ASCII text representation
            - 'latex': LaTeX/TikZ code for inclusion in documents
        filename: Optional file path to save the visualization. File extension
            should match the output format (.png/.pdf for matplotlib, .txt for text,
            .tex for latex).

    Returns:
        Visualization object: matplotlib.figure.Figure for 'matplotlib' format,
        string containing diagram for 'text' and 'latex' formats.

    Raises:
        ValueError: If an unknown output_format is specified.

    Example:
        >>> import numpy as np
        >>> from quantum_nn.circuits import ParameterizedCircuit
        >>> # Create and visualize a simple circuit
        >>> params = np.random.uniform(-np.pi, np.pi, 4)
        >>> fig = draw_circuit(my_circuit, params=params, style='blueprint')
        >>> # Save the visualization
        >>> draw_circuit(my_circuit, params=params, filename='circuit.png')
    """
    # Validate inputs
    if not isinstance(circuit, (ParameterizedCircuit, qml.QNode)):
        # Allow duck-typed objects that have the required interface
        if not (hasattr(circuit, "get_n_params") and hasattr(circuit, "_circuit_fn")):
            raise TypeError(
                "circuit must be either a ParameterizedCircuit or qml.QNode, "
                f"got {type(circuit).__name__}"
            )

    valid_styles = {"default", "blueprint", "sketch"}
    if style not in valid_styles:
        raise ValueError(f"style must be one of {valid_styles}, got '{style}'")

    valid_formats = {"matplotlib", "text", "latex"}
    if output_format.lower() not in valid_formats:
        raise ValueError(
            f"output_format must be one of {valid_formats}, got '{output_format}'"
        )

    # Extract QNode from ParameterizedCircuit if necessary
    qnode = circuit.circuit if hasattr(circuit, "circuit") else circuit

    # Set random parameters if not provided and circuit requires them
    if params is None and hasattr(circuit, "get_n_params"):
        # Generate random parameters in the range [-π, π] for visualization
        params = np.random.uniform(-np.pi, np.pi, size=circuit.get_n_params())

    # Create drawing function that executes the circuit for visualization
    def circuit_function(*args, **kwargs):
        if hasattr(circuit, "_circuit_fn"):
            # Execute the parameterized circuit with provided parameters and inputs
            circuit._circuit_fn(params, inputs)
        else:
            # For a raw QNode, execute with appropriate arguments
            args_list = []
            if params is not None:
                args_list.append(params)
            if inputs is not None:
                args_list.append(inputs)
            qnode.func(*args_list)

    # Generate visualization based on output format
    if output_format.lower() == "matplotlib":
        # Use PennyLane's matplotlib drawer for interactive figures
        fig, ax = plt.subplots(figsize=figsize)
        drawer = qml.draw_mpl(circuit_function)
        drawer(ax=ax)

        # Apply custom styling to the matplotlib figure
        if style == "blueprint":
            # Technical blueprint style with blue color scheme
            ax.set_facecolor("#f0f8ff")  # Alice blue background
            plt.setp(ax.spines.values(), color="navy")
            ax.tick_params(colors="navy")
            ax.set_title("Quantum Circuit Blueprint", color="navy")
        elif style == "sketch":
            # Hand-drawn sketch style with serif font
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = "Comic Sans MS"
            ax.set_title("Quantum Circuit Sketch")
        else:
            # Default professional style
            ax.set_title("Quantum Circuit Diagram")

        # Save figure if filename is provided
        if filename:
            plt.savefig(filename, bbox_inches="tight")

        return fig

    elif output_format.lower() == "text":
        # Use PennyLane's ASCII text drawer for console output
        text_diagram = qml.draw(circuit_function)()

        # Save text output if filename is provided
        if filename:
            with open(filename, "w") as f:
                f.write(text_diagram)

        return text_diagram

    elif output_format.lower() == "latex":
        # Use PennyLane's LaTeX/TikZ drawer for document inclusion
        latex_diagram = qml.draw_mpl(circuit_function, style="sketch")

        # Generate TikZ source code if .tex filename is provided
        if filename and filename.endswith(".tex"):
            tikz_code = qml.draw(
                circuit_function,
                expansion_strategy="device",
                output_format="latex_source",
            )()
            with open(filename, "w") as f:
                f.write(tikz_code)

        return latex_diagram

    else:
        # This should not be reachable due to input validation above
        raise ValueError(f"Unknown output format: {output_format}")
