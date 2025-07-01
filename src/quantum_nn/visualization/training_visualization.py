"""
Training visualization tools.

This module provides functions to visualize the training process and loss landscape
of quantum neural networks, helping with optimization analysis and debugging.
"""

from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Constants for visualization
DEFAULT_RESOLUTION = 50  # Default grid resolution for loss landscape
DEFAULT_CONTOUR_LEVELS = 50  # Number of contour levels in contour plots


def plot_loss_landscape(
    loss_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    param_indices: Tuple[int, int] = (0, 1),
    ranges: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-np.pi, np.pi),
        (-np.pi, np.pi),
    ),
    resolution: int = DEFAULT_RESOLUTION,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Quantum Loss Landscape",
    plot_type: str = "surface",
    colormap: str = "viridis",
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the loss landscape by varying two parameters while keeping others fixed.

    Creates a 2D slice of the loss function by varying two parameters across
    specified ranges, useful for understanding optimization challenges and
    local minima structure.

    Args:
        loss_fn: Function that computes loss given a parameter array. Should
            accept np.ndarray of shape matching params and return float.
        params: Current parameter values to use as the center point. Parameters
            not being varied will be held at these values.
        param_indices: Tuple of (index1, index2) specifying which two parameters
            to vary. Must be valid indices for the params array.
        ranges: Tuple of ((min1, max1), (min2, max2)) specifying the range of
            values to explore for each parameter.
        resolution: Number of grid points along each axis. Higher values give
            smoother plots but take longer to compute.
        figsize: Figure size as (width, height) in inches.
        title: Title for the plot.
        plot_type: Type of visualization. Options:
            - 'surface': 3D surface plot
            - 'contour': 2D contour plot
            - 'both': Both surface and contour in side-by-side subplots
        colormap: Matplotlib colormap name for color coding loss values.
        filename: Optional file path to save the plot.

    Returns:
        matplotlib.figure.Figure: The created figure object.

    Raises:
        ValueError: If param_indices contains invalid indices or plot_type
            is not recognized.

    Example:
        >>> def simple_loss(p):
        ...     return np.sum((p - 1.0) ** 2)  # Minimum at p = [1, 1, ...]
        >>> params = np.zeros(5)
        >>> fig = plot_loss_landscape(simple_loss, params, param_indices=(0, 1))
    """
    # Validate inputs
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")

    if len(param_indices) != 2:
        raise ValueError("param_indices must contain exactly 2 indices")

    max_idx = len(params) - 1
    if not all(0 <= idx <= max_idx for idx in param_indices):
        raise ValueError(
            f"param_indices must be in range [0, {max_idx}], got {param_indices}"
        )

    valid_plot_types = {"surface", "contour", "both"}
    if plot_type not in valid_plot_types:
        raise ValueError(
            f"plot_type must be one of {valid_plot_types}, got '{plot_type}'"
        )
    # Create parameter grid for exploration
    p1_range = np.linspace(ranges[0][0], ranges[0][1], resolution)
    p2_range = np.linspace(ranges[1][0], ranges[1][1], resolution)
    p1_grid, p2_grid = np.meshgrid(p1_range, p2_range)

    # Compute loss for each point in the grid
    loss_grid = np.zeros_like(p1_grid)
    for i in range(resolution):
        for j in range(resolution):
            # Create a copy of current parameters to avoid modifying original
            p = params.copy()

            # Update the two parameters we're varying
            p[param_indices[0]] = p1_grid[i, j]
            p[param_indices[1]] = p2_grid[i, j]

            # Compute loss at this parameter combination
            loss_grid[i, j] = loss_fn(p)

    # Create figure based on plot type
    if plot_type == "both":
        # Create side-by-side subplots for surface and contour
        fig = plt.figure(figsize=(figsize[0] * 2, figsize[1]))
        ax1 = fig.add_subplot(121, projection="3d")  # 3D surface
        ax2 = fig.add_subplot(122)  # 2D contour
    else:
        fig = plt.figure(figsize=figsize)
        if plot_type == "surface":
            ax1 = fig.add_subplot(111, projection="3d")
        else:  # contour
            ax2 = fig.add_subplot(111)

    # Plot 3D surface if requested
    if plot_type == "surface" or plot_type == "both":
        surf = ax1.plot_surface(
            p1_grid,
            p2_grid,
            loss_grid,
            cmap=colormap,
            linewidth=0,
            antialiased=True,
            alpha=0.8,
        )

        # Add color bar for loss values
        cbar = fig.colorbar(surf, ax=ax1, shrink=0.7, aspect=10)
        cbar.set_label("Loss")

        # Mark current parameter location in 3D
        current_loss = loss_fn(params)
        ax1.scatter(
            params[param_indices[0]],
            params[param_indices[1]],
            current_loss,
            color="red",
            s=100,
            marker="*",
            label="Current Parameters",
        )

        # Set axis labels
        ax1.set_xlabel(f"Parameter {param_indices[0]}")
        ax1.set_ylabel(f"Parameter {param_indices[1]}")
        ax1.set_zlabel("Loss")

        # Set subplot title
        if plot_type == "surface":
            ax1.set_title(title)
        else:
            ax1.set_title("3D Surface")

    # Plot 2D contour if requested
    if plot_type == "contour" or plot_type == "both":
        contour = ax2.contourf(
            p1_grid, p2_grid, loss_grid, levels=DEFAULT_CONTOUR_LEVELS, cmap=colormap
        )

        # Add color bar for loss values
        cbar = fig.colorbar(contour, ax=ax2)
        cbar.set_label("Loss")

        # Mark current parameter location in 2D
        ax2.scatter(
            params[param_indices[0]],
            params[param_indices[1]],
            color="red",
            s=100,
            marker="*",
            label="Current Parameters",
        )

        # Set axis labels
        ax2.set_xlabel(f"Parameter {param_indices[0]}")
        ax2.set_ylabel(f"Parameter {param_indices[1]}")

        # Set subplot title
        if plot_type == "contour":
            ax2.set_title(title)
        else:
            ax2.set_title("Contour Map")

    # Add legend to appropriate subplot
    if plot_type == "surface":
        ax1.legend()
    elif plot_type == "contour":
        ax2.legend()
    else:  # both
        ax2.legend()  # Legend on contour plot for both case

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches="tight")

    return fig


def plot_training_trajectory(
    history: dict,
    figsize: Tuple[int, int] = (10, 6),
    metrics: Optional[List[str]] = None,
    include_validation: bool = True,
    title: str = "Training History",
    legend_loc: str = "best",
    grid: bool = True,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training metrics over epochs from training history.

    Creates line plots showing how training and validation metrics evolve
    during the training process. Supports multiple metrics in separate subplots.

    Args:
        history: Training history dictionary containing metric values over epochs.
            Expected format: {'metric_name': [values], 'val_metric_name': [values]}.
            Common examples include {'loss': [...], 'accuracy': [...],
            'val_loss': [...], 'val_accuracy': [...]}.
        figsize: Figure size as (width, height) in inches.
        metrics: List of metric names to plot. If None, plots all available
            metrics (excluding validation metrics which are handled separately).
        include_validation: If True, includes validation metrics (prefixed with
            'val_') alongside training metrics when available.
        title: Overall title for the figure.
        legend_loc: Location for legends. Standard matplotlib location strings
            like 'best', 'upper right', 'lower left', etc.
        grid: If True, adds grid lines to subplots for easier reading.
        filename: Optional file path to save the plot.

    Returns:
        matplotlib.figure.Figure: The created figure object.

    Raises:
        ValueError: If history is empty or contains no valid metrics.

    Example:
        >>> history = {
        ...     'loss': [0.5, 0.3, 0.2, 0.1],
        ...     'accuracy': [0.7, 0.8, 0.9, 0.95],
        ...     'val_loss': [0.6, 0.4, 0.25, 0.15],
        ...     'val_accuracy': [0.65, 0.75, 0.85, 0.9]
        ... }
        >>> fig = plot_training_trajectory(history)
    """
    # Validate input
    if not history:
        raise ValueError("history dictionary cannot be empty")

    # Extract available metrics from history (excluding validation metrics)
    available_metrics = []
    for key in history.keys():
        if not key.startswith("val_") and len(history[key]) > 0:
            available_metrics.append(key)

    if not available_metrics:
        raise ValueError("No valid training metrics found in history")

    # Use all available metrics if none specified
    if metrics is None:
        metrics = available_metrics
    else:
        # Validate that requested metrics exist
        missing_metrics = [m for m in metrics if m not in available_metrics]
        if missing_metrics:
            raise ValueError(
                f"Requested metrics not found in history: {missing_metrics}"
            )

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)

    # Handle case with only one metric (axes is not a list)
    if len(metrics) == 1:
        axes = [axes]

    # Plot each metric in its own subplot
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Plot training metric if available
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(
                epochs, history[metric], "b-", linewidth=2, label=f"Training {metric}"
            )

        # Plot validation metric if available and requested
        val_metric = f"val_{metric}"
        if include_validation and val_metric in history:
            val_epochs = range(1, len(history[val_metric]) + 1)
            ax.plot(
                val_epochs,
                history[val_metric],
                "r--",
                linewidth=2,
                label=f"Validation {metric}",
            )

        # Configure subplot appearance
        ax.set_ylabel(metric.capitalize())

        # Add legend and grid
        ax.legend(loc=legend_loc)
        if grid:
            ax.grid(True, linestyle="--", alpha=0.7)

        # Set title for the first subplot only
        if i == 0:
            ax.set_title(title)

    # Set x-label for the bottom subplot
    axes[-1].set_xlabel("Epoch")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches="tight")

    return fig
