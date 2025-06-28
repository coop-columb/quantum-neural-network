"""
Quantum state visualization tools.

This module provides functions to visualize quantum states
using various representation formats.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pennylane as qml


def plot_state_amplitudes(
    state: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Quantum State Amplitudes",
    show_phase: bool = True,
    labels: Optional[List[str]] = None,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the amplitudes of a quantum state.

    Args:
        state: Quantum state vector
        figsize: Figure size
        title: Plot title
        show_phase: Whether to show the phase of amplitudes
        labels: Custom labels for basis states
        filename: File to save the plot to (optional)

    Returns:
        Matplotlib figure
    """
    n_states = len(state)
    n_qubits = int(np.log2(n_states))

    # Generate default labels if not provided
    if labels is None:
        labels = [format(i, f"0{n_qubits}b") for i in range(n_states)]

    # Compute amplitudes and phases
    amplitudes = np.abs(state)
    phases = np.angle(state)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot amplitudes as bar chart
    x = np.arange(n_states)
    bars = ax.bar(x, amplitudes, width=0.6)

    # Color bars according to phase if requested
    if show_phase:
        normalized_phases = (phases + np.pi) / (2 * np.pi)
        for i, bar in enumerate(bars):
            bar.set_color(cm.hsv(normalized_phases[i]))

        # Add color bar for phase reference
        sm = plt.cm.ScalarMappable(cmap=cm.hsv, norm=plt.Normalize(0, 2 * np.pi))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Phase (radians)")

    # Set labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Basis State")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

    # Adjust layout
    plt.tight_layout()

    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches="tight")

    return fig


def plot_state_bloch(
    state: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Quantum State Bloch Sphere Representation",
    qubit_indices: Optional[List[int]] = None,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the Bloch sphere representation of a quantum state.

    Args:
        state: Quantum state vector
        figsize: Figure size
        title: Plot title
        qubit_indices: Indices of qubits to visualize (defaults to all)
        filename: File to save the plot to (optional)

    Returns:
        Matplotlib figure
    """
    n_states = len(state)
    n_qubits = int(np.log2(n_states))

    # Default to all qubits if not specified
    if qubit_indices is None:
        qubit_indices = list(
            range(min(n_qubits, 6))
        )  # Limit to 6 qubits for readability

    # Calculate number of subplots
    n_plots = len(qubit_indices)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Create density matrix from state vector
    density_matrix = np.outer(state, np.conj(state))

    for i, qubit_idx in enumerate(qubit_indices):
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")

        # Calculate reduced density matrix for this qubit
        reduced_dm = qml.math.reduced_dm(density_matrix, [qubit_idx], n_qubits)

        # Calculate Bloch vector
        x = np.real(reduced_dm[0, 1] + reduced_dm[1, 0])
        y = np.imag(reduced_dm[0, 1] - reduced_dm[1, 0])
        z = np.real(reduced_dm[0, 0] - reduced_dm[1, 1])

        # Draw Bloch sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        sphere_x = 0.98 * np.outer(np.cos(u), np.sin(v))
        sphere_y = 0.98 * np.outer(np.sin(u), np.sin(v))
        sphere_z = 0.98 * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(sphere_x, sphere_y, sphere_z, color="lightgray", alpha=0.2)

        # Draw coordinate axes
        ax.plot([-1, 1], [0, 0], [0, 0], "k--", alpha=0.5, lw=1)
        ax.plot([0, 0], [-1, 1], [0, 0], "k--", alpha=0.5, lw=1)
        ax.plot([0, 0], [0, 0], [-1, 1], "k--", alpha=0.5, lw=1)

        # Draw Bloch vector
        ax.quiver(0, 0, 0, x, y, z, color="r", arrow_length_ratio=0.1, lw=2)

        # Set labels
        ax.text(1.1, 0, 0, r"$x$", fontsize=12)
        ax.text(0, 1.1, 0, r"$y$", fontsize=12)
        ax.text(0, 0, 1.1, r"$z$", fontsize=12)

        # Set title for this subplot
        ax.set_title(f"Qubit {qubit_idx}")

        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)

    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches="tight")

    return fig


def plot_state_city(
    state: np.ndarray,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Quantum State City Plot",
    colormap: str = "viridis",
    show_wireframe: bool = True,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Create a 3D city plot of a quantum state.

    Args:
        state: Quantum state vector
        figsize: Figure size
        title: Plot title
        colormap: Matplotlib colormap name
        show_wireframe: Whether to show wireframe
        filename: File to save the plot to (optional)

    Returns:
        Matplotlib figure
    """
    n_states = len(state)
    n_qubits = int(np.log2(n_states))

    # Reshape state for 2D visualization if possible
    if n_qubits % 2 == 0:
        # Even number of qubits, make a square grid
        grid_size = 2 ** (n_qubits // 2)
        amplitudes = np.abs(state).reshape(grid_size, grid_size)
        phases = np.angle(state).reshape(grid_size, grid_size)
    else:
        # Odd number of qubits, make a rectangular grid
        grid_width = 2 ** ((n_qubits + 1) // 2)
        grid_height = 2 ** ((n_qubits - 1) // 2)
        amplitudes = np.abs(state).reshape(grid_height, grid_width)
        phases = np.angle(state).reshape(grid_height, grid_width)

    # Create meshgrid for plotting
    x = np.arange(amplitudes.shape[1])
    y = np.arange(amplitudes.shape[0])
    x, y = np.meshgrid(x, y)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Create city plot
    cmap = plt.get_cmap(colormap)
    normalized_phases = (phases + np.pi) / (2 * np.pi)
    colors = cmap(normalized_phases)

    # Plot as 3D bars
    dx = dy = 0.8
    for i in range(amplitudes.shape[0]):
        for j in range(amplitudes.shape[1]):
            if amplitudes[i, j] > 1e-6:  # Only plot non-zero amplitudes
                ax.bar3d(
                    x[i, j] - dx / 2,
                    y[i, j] - dy / 2,
                    0,
                    dx,
                    dy,
                    amplitudes[i, j],
                    color=colors[i, j],
                    shade=True,
                )

    # Add wireframe if requested
    if show_wireframe:
        ax.plot_wireframe(x, y, amplitudes, color="black", alpha=0.3, linewidth=0.5)

    # Set labels
    ax.set_xlabel("Basis State (First Half)")
    ax.set_ylabel("Basis State (Second Half)")
    ax.set_zlabel("Amplitude")
    ax.set_title(title)

    # Add color bar for phase reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 2 * np.pi))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Phase (radians)")

    # Set axis limits and ticks
    ax.set_xlim(-0.5, amplitudes.shape[1] - 0.5)
    ax.set_ylim(-0.5, amplitudes.shape[0] - 0.5)
    ax.set_zlim(0, np.max(amplitudes) * 1.1)

    # Set tick labels
    if amplitudes.shape[0] <= 16 and amplitudes.shape[1] <= 16:
        x_labels = [format(i, f"0{n_qubits//2}b") for i in range(amplitudes.shape[1])]
        y_labels = [
            format(i, f"0{n_qubits-n_qubits//2}b") for i in range(amplitudes.shape[0])
        ]
        ax.set_xticks(np.arange(amplitudes.shape[1]))
        ax.set_yticks(np.arange(amplitudes.shape[0]))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

    # Adjust view angle for better visualization
    ax.view_init(elev=30, azim=45)

    # Adjust layout
    plt.tight_layout()

    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches="tight")

    return fig
