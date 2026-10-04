"""NeuroFence UI package.

Exposes the two main UI components:
  - HeatmapWidget: matplotlib-based activation heatmap embedded in a QWidget
  - MainWindow: full application window
"""

from .heatmap_widget import HeatmapWidget
from .main_window import MainWindow

__all__ = ["HeatmapWidget", "MainWindow"]
