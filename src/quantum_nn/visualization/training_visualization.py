"""
Training visualization tools.

This module provides functions to visualize the training process
and loss landscape of quantum neural networks.
"""

from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_loss_landscape(
    loss_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    param_indices: Tuple[int, int] = (0, 1),
    ranges: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-np.pi, np.pi),
        (-np.pi, np.pi),
    ),
    resolution: int = 50,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Quantum Loss Landscape",
    plot_type: str = "surface",
    colormap: str = "viridis",
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the loss landscape by varying two parameters.

    Args:
        loss_fn: Function that computes loss given parameters
        params: Current parameters
        param_indices: Indices of parameters to vary
        ranges: Range of parameter values to explore
        resolution: Number of points in each dimension
        figsize: Figure size
        title: Plot title
        plot_type: Type of plot ('surface', 'contour', or 'both')
        colormap: Matplotlib colormap name
        filename: File to save the plot to (optional)

    Returns:
        Matplotlib figure
    """
    # Create parameter grid
    p1_range = np.linspace(ranges[0][0], ranges[0][1], resolution)
    p2_range = np.linspace(ranges[1][0], ranges[1][1], resolution)
    p1_grid, p2_grid = np.meshgrid(p1_range, p2_range)

    # Compute loss for each point in the grid
    loss_grid = np.zeros_like(p1_grid)
    for i in range(resolution):
        for j in range(resolution):
            # Create a copy of current parameters
            p = params.copy()

            # Update the two parameters we're varying
            p[param_indices[0]] = p1_grid[i, j]
            p[param_indices[1]] = p2_grid[i, j]

            # Compute loss
            loss_grid[i, j] = loss_fn(p)

    # Create figure
    if plot_type == "both":
        fig = plt.figure(figsize=(figsize[0] * 2, figsize[1]))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122)
    else:
        fig = plt.figure(figsize=figsize)
        if plot_type == "surface":
            ax1 = fig.add_subplot(111, projection="3d")
        else:  # contour
            ax2 = fig.add_subplot(111)

    # Plot surface
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

        # Add color bar
        cbar = fig.colorbar(surf, ax=ax1, shrink=0.7, aspect=10)
        cbar.set_label("Loss")

        # Mark current parameters
        ax1.scatter(
            params[param_indices[0]],
            params[param_indices[1]],
            loss_fn(params),
            color="red",
            marker="*",
            label="Current Parameters",
            s=100,
        )

        # Set labels
        ax1.set_xlabel(f"Parameter {param_indices[0]}")
        ax1.set_ylabel(f"Parameter {param_indices[1]}")
        ax1.set_zlabel("Loss")

        if plot_type == "surface":
            ax1.set_title(title)
        else:
            ax1.set_title("3D Surface")

    # Plot contour
    if plot_type == "contour" or plot_type == "both":
        contour = ax2.contourf(p1_grid, p2_grid, loss_grid, levels=50, cmap=colormap)

        # Add color bar
        cbar = fig.colorbar(contour, ax=ax2)
        cbar.set_label("Loss")

        # Mark current parameters
        ax2.scatter(
            params[param_indices[0]],
            params[param_indices[1]],
            color="red",
            s=100,
            marker="*",
            label="Current Parameters",
        )

        # Set labels
        ax2.set_xlabel(f"Parameter {param_indices[0]}")
        ax2.set_ylabel(f"Parameter {param_indices[1]}")

        if plot_type == "contour":
            ax2.set_title(title)
        else:
            ax2.set_title("Contour Map")

    # Add legend
    if plot_type == "surface":
        ax1.legend()
    elif plot_type == "contour":
        ax2.legend()
    else:
        ax2.legend()

    # Adjust layout
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
    Plot training metrics over epochs.

    Args:
        history: Training history dictionary (as returned by model.fit)
        figsize: Figure size
        metrics: Metrics to plot (defaults to all)
        include_validation: Whether to include validation metrics
        title: Plot title
        legend_loc: Location of the legend
        grid: Whether to show grid
        filename: File to save the plot to (optional)

    Returns:
        Matplotlib figure
    """
    # Extract available metrics from history
    available_metrics = []
    for key in history.keys():
        if not key.startswith("val_"):
            available_metrics.append(key)

    # Use all available metrics if none specified
    if metrics is None:
        metrics = available_metrics

    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)

    # Handle case with only one metric
    if len(metrics) == 1:
        axes = [axes]

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Plot training metric
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], "b-", label=f"Training {metric}")

        # Plot validation metric if available and requested
        val_metric = f"val_{metric}"
        if include_validation and val_metric in history:
            epochs = range(1, len(history[val_metric]) + 1)
            ax.plot(epochs, history[val_metric], "r-", label=f"Validation {metric}")

        # Set labels
        ax.set_ylabel(metric.capitalize())

        # Add legend and grid
        ax.legend(loc=legend_loc)
        if grid:
            ax.grid(True, linestyle="--", alpha=0.7)

        # Set title for the first subplot
        if i == 0:
            ax.set_title(title)

    # Set x-label for the bottom subplot
    axes[-1].set_xlabel("Epoch")

    # Adjust layout
    plt.tight_layout()

    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches="tight")

    return fig
