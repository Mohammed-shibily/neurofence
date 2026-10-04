"""NeuroFence desktop application entry point.

Launches the PyQt5 forensic GUI for offline GPT-2 backdoor-neuron analysis.

Usage
-----
    python main.py

The application runs entirely offline (``local_files_only=True`` via
``load_model_sandboxed``) and never accesses the network or any cloud API.
"""

import sys
from pathlib import Path

# Ensure the project root is importable when launched as ``python main.py``
# (or via an absolute path) before importing application packages.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PyQt5.QtGui import QColor, QPalette  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from src.ui.main_window import MainWindow  # noqa: E402

# ---------------------------------------------------------------------------
# Dark theme (deepest #0d1117 / panels #161b22 / widgets #21262d)
# ---------------------------------------------------------------------------
_DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #e6edf3;
}
QLabel {
    color: #e6edf3;
    background-color: transparent;
}
QPushButton {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
}
QPushButton:hover { border-color: #58a6ff; color: #58a6ff; }
QPushButton:pressed { background-color: #30363d; }
QPushButton:disabled { color: #8b949e; background-color: #161b22; border-color: #30363d; }
QTableWidget {
    background-color: #161b22;
    alternate-background-color: #21262d;
    color: #e6edf3;
    gridline-color: #30363d;
    border: 1px solid #30363d;
}
QTableWidget::item { padding: 4px 6px; }
QTableWidget::item:selected { background-color: #58a6ff; color: #0d1117; }
QHeaderView::section {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    padding: 6px;
    font-weight: bold;
}
QTabWidget::pane {
    border: 1px solid #30363d;
    background-color: #161b22;
    top: -1px;
}
QTabBar::tab {
    background-color: #161b22;
    color: #8b949e;
    border: 1px solid #30363d;
    border-bottom: none;
    padding: 8px 18px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #21262d;
    color: #e6edf3;
    border-top: 2px solid #58a6ff;
}
QTabBar::tab:hover { color: #e6edf3; }
QTextEdit {
    background-color: #161b22;
    color: #e6edf3;
    border: 1px solid #30363d;
}
QProgressBar {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    color: #e6edf3;
    text-align: center;
}
QProgressBar::chunk { background-color: #58a6ff; border-radius: 3px; }
QLineEdit {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: #58a6ff;
}
QScrollBar:vertical { background: #0d1117; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #30363d; min-height: 24px; border-radius: 6px; }
QScrollBar::handle:vertical:hover { background: #58a6ff; }
QScrollBar:horizontal { background: #0d1117; height: 12px; margin: 0; }
QScrollBar::handle:horizontal { background: #30363d; min-width: 24px; border-radius: 6px; }
QScrollBar::handle:horizontal:hover { background: #58a6ff; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
QToolTip {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    padding: 4px;
}
QMessageBox { background-color: #161b22; }
QMessageBox QLabel { color: #e6edf3; }
"""


def _apply_dark_palette(app: QApplication) -> None:
    """Apply a dark QPalette so native dialogs match the stylesheet."""
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#0d1117"))
    palette.setColor(QPalette.WindowText, QColor("#e6edf3"))
    palette.setColor(QPalette.Base, QColor("#161b22"))
    palette.setColor(QPalette.AlternateBase, QColor("#21262d"))
    palette.setColor(QPalette.ToolTipBase, QColor("#21262d"))
    palette.setColor(QPalette.ToolTipText, QColor("#e6edf3"))
    palette.setColor(QPalette.Text, QColor("#e6edf3"))
    palette.setColor(QPalette.Button, QColor("#21262d"))
    palette.setColor(QPalette.ButtonText, QColor("#e6edf3"))
    palette.setColor(QPalette.Highlight, QColor("#58a6ff"))
    palette.setColor(QPalette.HighlightedText, QColor("#0d1117"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#8b949e"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#8b949e"))
    app.setPalette(palette)


def main() -> int:
    """Create the application, apply the dark theme, and run the event loop.

    Returns:
        The Qt event-loop exit code.
    """
    app = QApplication(sys.argv)
    app.setApplicationName("NeuroFence")
    app.setApplicationDisplayName("NeuroFence - Model Backdoor Forensics")
    app.setStyle("Fusion")
    app.setStyleSheet(_DARK_STYLESHEET)
    _apply_dark_palette(app)

    window = MainWindow()
    window.resize(1000, 700)
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())