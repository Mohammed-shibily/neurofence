"""HeatmapWidget — matplotlib activation heatmap embedded in a PyQt5 QWidget.

Displays the 6 × 3072 max-exceedance-margin matrix returned by
``scan_model()["heatmap"]`` as an ``imshow`` plot on a dark background
using the ``inferno`` colormap.

Public interface
----------------
``set_data(heatmap)``
    Accept a (6, 3072) numpy array (or list-of-lists) and re-render the plot.
``clear()``
    Reset to a blank placeholder state.
"""

from typing import Optional, Union

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

# ---------------------------------------------------------------------------
# Colour palette (must match the app-wide dark theme)
# ---------------------------------------------------------------------------
_BG_DEEP = "#1a1a2e"
_BG_PANEL = "#16213e"
_TEXT_PRIMARY = "#e6edf3"
_TEXT_SECONDARY = "#8b949e"
_TICK_COLOR = "#ffffff"
_ACCENT = "#58a6ff"


class HeatmapWidget(QWidget):
    """A self-contained QWidget that embeds a matplotlib heatmap figure.

    The figure uses a dark background consistent with the application theme
    and the ``inferno`` colormap so that high exceedance-margin neurons stand
    out in bright yellow/white against a dark purple background.

    Usage::

        widget = HeatmapWidget(parent)
        widget.set_data(np.array(result["heatmap"]))   # shape (6, 3072)
        widget.clear()   # reset to placeholder

    Parameters
    ----------
    parent:
        Optional parent QWidget.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Build the figure with a dark background
        self._fig = Figure(facecolor=_BG_DEEP, tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._ax = None
        self._cbar = None
        self._im = None

        self._init_axes()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_axes(self) -> None:
        """Create the axes and draw the initial placeholder."""
        self._fig.clear()
        self._ax = self._fig.add_subplot(111)
        self._style_axes()
        self._draw_placeholder()
        self._canvas.draw_idle()

    def _style_axes(self) -> None:
        """Apply dark-theme styling to the current axes."""
        ax = self._ax
        ax.set_facecolor(_BG_PANEL)
        ax.tick_params(colors=_TICK_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.xaxis.label.set_color(_TEXT_PRIMARY)
        ax.yaxis.label.set_color(_TEXT_PRIMARY)
        ax.title.set_color(_TEXT_PRIMARY)

    def _draw_placeholder(self) -> None:
        """Render a placeholder message when no data is loaded."""
        ax = self._ax
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(
            0.5, 0.5,
            "Run a scan to view the activation heatmap",
            ha="center", va="center",
            color=_TEXT_SECONDARY,
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            "Activation Heatmap - Max Exceedance Margin per Neuron",
            color=_TEXT_PRIMARY,
            fontsize=11,
            pad=10,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_data(self, heatmap: Union[np.ndarray, list]) -> None:
        """Render the heatmap from a (n_layers × n_neurons) matrix.

        Clears the previous plot completely and redraws with the new data,
        including a labelled colorbar.

        Args:
            heatmap: A (6, 3072) array-like of float32 exceedance margins.
                     Values are sigma units above the clean-model baseline
                     maximum activation.
        """
        data = np.asarray(heatmap, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(
                f"heatmap must be 2-D (n_layers, n_neurons); got shape {data.shape}."
            )

        self._fig.clear()
        self._ax = self._fig.add_subplot(111)
        self._style_axes()

        n_layers, n_neurons = data.shape
        vmax = float(np.max(data))
        vmin = 0.0

        self._im = self._ax.imshow(
            data,
            aspect="auto",
            cmap="inferno",
            interpolation="nearest",
            vmin=vmin,
            vmax=max(vmax, 1.0),   # guard against all-zero heatmap
            origin="upper",
        )

        # Colorbar
        self._cbar = self._fig.colorbar(self._im, ax=self._ax, pad=0.02)
        self._cbar.set_label(
            "Exceedance Margin (sigma above clean max)",
            color=_TEXT_PRIMARY,
            fontsize=9,
        )
        self._cbar.ax.yaxis.set_tick_params(color=_TICK_COLOR)
        self._cbar.ax.tick_params(labelcolor=_TICK_COLOR, labelsize=8)
        self._cbar.outline.set_edgecolor("#30363d")

        # Axes labels and ticks
        self._ax.set_xlabel(f"Neuron (0-{n_neurons - 1})", fontsize=9)
        self._ax.set_ylabel(f"Layer (0-{n_layers - 1})", fontsize=9)
        self._ax.set_yticks(range(n_layers))
        self._ax.set_yticklabels(
            [f"Layer {i}" for i in range(n_layers)],
            fontsize=8,
            color=_TICK_COLOR,
        )
        self._ax.set_title(
            "Activation Heatmap - Max Exceedance Margin per Neuron",
            color=_TEXT_PRIMARY,
            fontsize=11,
            pad=10,
        )

        # Annotate global maximum
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        self._ax.scatter(
            [max_idx[1]], [max_idx[0]],
            s=60, c="#58a6ff", marker="x", linewidths=1.5,
            zorder=5, label=f"Peak  L{max_idx[0]} N{max_idx[1]}  ({vmax:.1f}σ)",
        )
        legend = self._ax.legend(
            loc="upper right", fontsize=7,
            facecolor=_BG_PANEL, edgecolor="#30363d",
            labelcolor=_TEXT_PRIMARY,
        )

        self._canvas.draw_idle()

    def clear(self) -> None:
        """Reset the widget to its placeholder state."""
        self._im = None
        self._cbar = None
        self._init_axes()
