"""
Quantum state visualization tools.

This module provides functions to visualize quantum states using various
representation formats including amplitude plots, Bloch sphere representations,
and 3D city plots.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from mpl_toolkits.mplot3d import Axes3D

# Constants for visualization
AMPLITUDE_THRESHOLD = 1e-6  # Minimum amplitude to display in visualizations
MAX_QUBITS_BLOCH = 6  # Maximum number of qubits for Bloch sphere visualization
DEFAULT_SPHERE_ALPHA = 0.2  # Transparency for Bloch sphere surface
DEFAULT_WIREFRAME_ALPHA = 0.3  # Transparency for wireframe overlays


def plot_state_amplitudes(
    state: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Quantum State Amplitudes",
    show_phase: bool = True,
    labels: Optional[List[str]] = None,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the amplitudes and phases of a quantum state as a bar chart.

    Creates a bar chart visualization where the height represents the amplitude
    magnitude and the color represents the phase (if enabled).

    Args:
        state: Complex-valued quantum state vector of shape (2^n,) where n is
            the number of qubits. Must be normalized.
        figsize: Figure size as (width, height) in inches.
        title: Title for the plot.
        show_phase: If True, color-codes bars according to phase using HSV colormap.
            If False, uses uniform blue color.
        labels: Custom labels for basis states. If None, generates binary labels
            like '00', '01', '10', '11' for 2-qubit states.
        filename: Optional file path to save the plot. Supports standard matplotlib
            formats (.png, .pdf, .svg, etc.).

    Returns:
        matplotlib.figure.Figure: The created figure object.

    Raises:
        ValueError: If state vector length is not a power of 2.

    Example:
        >>> import numpy as np
        >>> # Create a simple superposition state |00⟩ + |11⟩
        >>> state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        >>> fig = plot_state_amplitudes(state, show_phase=True)
    """
    # Validate input
    n_states = len(state)
    if n_states == 0 or (n_states & (n_states - 1)) != 0:
        raise ValueError(f"State vector length must be a power of 2, got {n_states}")

    n_qubits = int(np.log2(n_states))

    # Generate default labels if not provided (binary representation)
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
        # Normalize phases from [-π, π] to [0, 1] for HSV colormap
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

    # Adjust layout to prevent label cutoff
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
    Plot individual qubits of a quantum state on Bloch spheres.

    Creates a visualization showing the reduced density matrix of each qubit
    as a point on the Bloch sphere. Each qubit gets its own 3D subplot.

    Args:
        state: Complex-valued quantum state vector of shape (2^n,) where n is
            the number of qubits. Must be normalized.
        figsize: Figure size as (width, height) in inches.
        title: Overall title for the figure.
        qubit_indices: List of qubit indices to visualize. If None, visualizes
            all qubits up to MAX_QUBITS_BLOCH for readability.
        filename: Optional file path to save the plot.

    Returns:
        matplotlib.figure.Figure: The created figure object.

    Raises:
        ValueError: If state vector length is not a power of 2, or if any
            qubit index is out of range.

    Note:
        For highly entangled states, individual qubit representations may appear
        near the center of the Bloch sphere, indicating mixed states.

    Example:
        >>> import numpy as np
        >>> # Create a Bell state |00⟩ + |11⟩
        >>> state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        >>> fig = plot_state_bloch(state, qubit_indices=[0, 1])
    """
    # Validate input
    n_states = len(state)
    if n_states == 0 or (n_states & (n_states - 1)) != 0:
        raise ValueError(f"State vector length must be a power of 2, got {n_states}")

    n_qubits = int(np.log2(n_states))

    # Default to all qubits if not specified, but limit for readability
    if qubit_indices is None:
        qubit_indices = list(range(min(n_qubits, MAX_QUBITS_BLOCH)))

    # Validate qubit indices
    if not all(0 <= idx < n_qubits for idx in qubit_indices):
        raise ValueError(
            f"All qubit indices must be in range [0, {n_qubits-1}], "
            f"got {qubit_indices}"
        )

    # Calculate grid layout for subplots
    n_plots = len(qubit_indices)
    n_cols = min(3, n_plots)  # Maximum 3 columns for better readability
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Create density matrix from state vector
    density_matrix = np.outer(state, np.conj(state))

    for i, qubit_idx in enumerate(qubit_indices):
        # Create 3D subplot for this qubit's Bloch sphere
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")

        # Calculate reduced density matrix for this specific qubit
        # Use a more compatible approach for reduced density matrix calculation
        n_states = len(state)

        # Manually calculate reduced density matrix
        reduced_dm = np.zeros((2, 2), dtype=complex)

        # Sum over all states where this qubit has value 0 or 1
        for i in range(n_states):
            for j in range(n_states):
                # Get binary representations
                i_bits = format(i, f"0{n_qubits}b")
                j_bits = format(j, f"0{n_qubits}b")

                # Check if all qubits except the target match
                i_other = i_bits[:qubit_idx] + i_bits[qubit_idx + 1 :]
                j_other = j_bits[:qubit_idx] + j_bits[qubit_idx + 1 :]

                if i_other == j_other:
                    # Get the target qubit values
                    i_target = int(i_bits[qubit_idx])
                    j_target = int(j_bits[qubit_idx])

                    # Add to reduced density matrix
                    reduced_dm[i_target, j_target] += state[i] * np.conj(state[j])

        # Calculate Bloch vector components from Pauli expectation values
        # x = ⟨σ_x⟩, y = ⟨σ_y⟩, z = ⟨σ_z⟩
        x = np.real(reduced_dm[0, 1] + reduced_dm[1, 0])  # ⟨σ_x⟩
        y = np.imag(reduced_dm[0, 1] - reduced_dm[1, 0])  # ⟨σ_y⟩
        z = np.real(reduced_dm[0, 0] - reduced_dm[1, 1])  # ⟨σ_z⟩

        # Draw semi-transparent Bloch sphere surface
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        # Scale slightly smaller than unit sphere for visual clarity
        sphere_radius = 0.98
        sphere_x = sphere_radius * np.outer(np.cos(u), np.sin(v))
        sphere_y = sphere_radius * np.outer(np.sin(u), np.sin(v))
        sphere_z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(
            sphere_x, sphere_y, sphere_z, color="lightgray", alpha=DEFAULT_SPHERE_ALPHA
        )

        # Draw coordinate axes for reference
        axis_length = 1.0
        ax.plot([-axis_length, axis_length], [0, 0], [0, 0], "k--", alpha=0.5, lw=1)
        ax.plot([0, 0], [-axis_length, axis_length], [0, 0], "k--", alpha=0.5, lw=1)
        ax.plot([0, 0], [0, 0], [-axis_length, axis_length], "k--", alpha=0.5, lw=1)

        # Draw Bloch vector as red arrow
        ax.quiver(0, 0, 0, x, y, z, color="r", arrow_length_ratio=0.1, lw=2)

        # Add axis labels
        ax.text(1.1, 0, 0, r"$X$", fontsize=12)
        ax.text(0, 1.1, 0, r"$Y$", fontsize=12)
        ax.text(0, 0, 1.1, r"$Z$", fontsize=12)

        # Set subplot title
        ax.set_title(f"Qubit {qubit_idx}")

        # Remove axis ticks for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Set equal aspect ratio for proper sphere appearance
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])

    # Adjust layout to prevent subplot overlap
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Leave space for suptitle

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
    Create a 3D city plot visualization of a quantum state.

    Displays quantum state amplitudes as 3D bars (buildings) arranged in a 2D grid,
    where bar height represents amplitude magnitude and color represents phase.

    Args:
        state: Complex-valued quantum state vector of shape (2^n,) where n is
            the number of qubits. Must be normalized.
        figsize: Figure size as (width, height) in inches.
        title: Title for the plot.
        colormap: Matplotlib colormap name for phase visualization. Good options
            include 'viridis', 'hsv', 'plasma', 'coolwarm'.
        show_wireframe: If True, overlays a wireframe grid for better depth perception.
        filename: Optional file path to save the plot.

    Returns:
        matplotlib.figure.Figure: The created figure object.

    Raises:
        ValueError: If state vector length is not a power of 2.

    Note:
        For states with many qubits, the grid layout attempts to create a roughly
        square arrangement. Even qubit numbers create perfect squares, while odd
        qubit numbers create rectangular grids.

    Example:
        >>> import numpy as np
        >>> # Create a 3-qubit GHZ state |000⟩ + |111⟩
        >>> state = np.zeros(8)
        >>> state[0] = state[7] = 1/np.sqrt(2)
        >>> fig = plot_state_city(state, colormap='hsv')
    """
    # Validate input
    n_states = len(state)
    if n_states == 0 or (n_states & (n_states - 1)) != 0:
        raise ValueError(f"State vector length must be a power of 2, got {n_states}")

    n_qubits = int(np.log2(n_states))

    # Reshape state for 2D grid visualization
    if n_qubits % 2 == 0:
        # Even number of qubits: create square grid
        grid_size = 2 ** (n_qubits // 2)
        amplitudes = np.abs(state).reshape(grid_size, grid_size)
        phases = np.angle(state).reshape(grid_size, grid_size)
    else:
        # Odd number of qubits: create rectangular grid
        grid_width = 2 ** ((n_qubits + 1) // 2)
        grid_height = 2 ** ((n_qubits - 1) // 2)
        amplitudes = np.abs(state).reshape(grid_height, grid_width)
        phases = np.angle(state).reshape(grid_height, grid_width)

    # Create coordinate meshgrid for 3D plotting
    x = np.arange(amplitudes.shape[1])
    y = np.arange(amplitudes.shape[0])
    x, y = np.meshgrid(x, y)

    # Create figure with 3D projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Setup colormap and phase normalization
    cmap = plt.get_cmap(colormap)
    # Normalize phases from [-π, π] to [0, 1] for colormap
    normalized_phases = (phases + np.pi) / (2 * np.pi)
    colors = cmap(normalized_phases)

    # Plot 3D bars (city buildings) for non-zero amplitudes
    bar_width = bar_depth = 0.8  # Bar dimensions for good visual separation
    for i in range(amplitudes.shape[0]):
        for j in range(amplitudes.shape[1]):
            # Only plot bars for amplitudes above threshold to reduce clutter
            if amplitudes[i, j] > AMPLITUDE_THRESHOLD:
                ax.bar3d(
                    x[i, j] - bar_width / 2,  # x position
                    y[i, j] - bar_depth / 2,  # y position
                    0,  # z bottom (always 0)
                    bar_width,  # width
                    bar_depth,  # depth
                    amplitudes[i, j],  # height (amplitude)
                    color=colors[i, j],  # color (phase)
                    shade=True,  # enable shading
                )

    # Add wireframe overlay for better depth perception
    if show_wireframe:
        ax.plot_wireframe(
            x,
            y,
            amplitudes,
            color="black",
            alpha=DEFAULT_WIREFRAME_ALPHA,
            linewidth=0.5,
        )

    # Set axis labels and title
    ax.set_xlabel("Basis State (First Half)")
    ax.set_ylabel("Basis State (Second Half)")
    ax.set_zlabel("Amplitude")
    ax.set_title(title)

    # Add color bar for phase reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 2 * np.pi))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20)
    cbar.set_label("Phase (radians)")

    # Set axis limits with small padding
    padding = 0.5
    ax.set_xlim(-padding, amplitudes.shape[1] - 1 + padding)
    ax.set_ylim(-padding, amplitudes.shape[0] - 1 + padding)
    ax.set_zlim(0, np.max(amplitudes) * 1.1)

    # Set tick labels for smaller grids (readability limit)
    max_ticks = 16
    if amplitudes.shape[0] <= max_ticks and amplitudes.shape[1] <= max_ticks:
        # Generate binary labels for grid positions
        x_labels = [format(i, f"0{n_qubits//2}b") for i in range(amplitudes.shape[1])]
        y_labels = [
            format(i, f"0{n_qubits-n_qubits//2}b") for i in range(amplitudes.shape[0])
        ]
        ax.set_xticks(np.arange(amplitudes.shape[1]))
        ax.set_yticks(np.arange(amplitudes.shape[0]))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

    # Set optimal viewing angle for 3D visualization
    ax.view_init(elev=30, azim=45)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches="tight")

    return fig
