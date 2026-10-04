"""NeuroFence main application window.

Provides:
  - ``MainWindow`` (QMainWindow): the full forensic desktop UI
  - ``ScanWorker`` (QThread): runs ``scan_model()`` off the GUI thread

Design constraints
------------------
* Never reads, imports, or displays ``data/poison_ground_truth.json``.
* All model loading goes through ``load_model_sandboxed()`` (local_files_only=True).
* ``scan_model()`` always runs inside ``ScanWorker`` — the GUI thread is
  never blocked by PyTorch forward passes.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.detection.analyzer import scan_model
from src.sandbox.loader import load_model_sandboxed
from src.ui.heatmap_widget import HeatmapWidget

# ---------------------------------------------------------------------------
# Path resolution — resolved once at import time, relative to this file
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # src/ui/
_PROJECT_ROOT = _HERE.parent.parent              # neurofence/
_BASELINE_DEFAULT = _PROJECT_ROOT / "data" / "baseline_clean.npz"

# Human-readable baseline reference shown in the UI so results are never
# misleading if NeuroFence is later pointed at an unrelated architecture.
_BASELINE_REL = "data/baseline_clean.npz"
_BASELINE_ARCH = "distilgpt2"


# ---------------------------------------------------------------------------
# Colour constants (mirrors heatmap_widget.py)
# ---------------------------------------------------------------------------
_C_BG_DEEP    = "#0d1117"
_C_BG_PANEL   = "#161b22"
_C_BG_WIDGET  = "#21262d"
_C_ACCENT     = "#58a6ff"
_C_DANGER     = "#f85149"
_C_SAFE       = "#3fb950"
_C_TEXT       = "#e6edf3"
_C_TEXT_DIM   = "#8b949e"
_C_BORDER     = "#30363d"


# ===========================================================================
# Background worker thread
# ===========================================================================

class ScanWorker(QThread):
    """Runs ``scan_model()`` in a background thread.

    Emits ``scan_complete`` with the result dict on success, or ``scan_error``
    with a human-readable error string on any exception.  The ``finished``
    signal (inherited from QThread) fires in both cases.

    Parameters
    ----------
    model_dir:
        Path to the model directory to scan.
    baseline_path:
        Path to ``baseline_clean.npz``.
    parent:
        Optional parent QObject.
    """

    scan_complete: pyqtSignal = pyqtSignal(dict)
    scan_error:    pyqtSignal = pyqtSignal(str)

    def __init__(
        self,
        model_dir: str,
        baseline_path: str,
        parent: Optional[QThread] = None,
    ) -> None:
        super().__init__(parent)
        self._model_dir = model_dir
        self._baseline_path = baseline_path

    def run(self) -> None:  # noqa: D102
        try:
            result = scan_model(
                model_dir=self._model_dir,
                baseline_path=self._baseline_path,
                progress=False,  # suppress stdout noise during GUI scan
            )
            self.scan_complete.emit(result)
        except Exception:  # noqa: BLE001
            msg = traceback.format_exc()
            self.scan_error.emit(msg)


# ===========================================================================
# Main application window
# ===========================================================================

class MainWindow(QMainWindow):
    """NeuroFence forensic desktop application main window.

    Layout::

        ┌─────────────────────────────────────────────────┐
        │  Top bar: [Browse] [model path] [Run Scan]      │
        │           [Export Report]  [progress bar]       │
        ├─────────────────────────────────────────────────┤
        │  Metadata panel (hash / arch / layers / vocab)  │
        ├─────────────────────────────────────────────────┤
        │  Tabs:  Summary | Flagged Neurons | Heatmap     │
        │         | Limitations                           │
        └─────────────────────────────────────────────────┘
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NeuroFence — Model Backdoor Forensics")
        self.setMinimumSize(900, 620)

        # Internal state
        self._model_dir: Optional[str] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._worker: Optional[ScanWorker] = None
        self._last_result: Optional[Dict[str, Any]] = None

        self._build_ui()
        self._set_initial_state()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(0)
        root_layout.setContentsMargins(0, 0, 0, 0)

        root_layout.addWidget(self._build_top_bar())
        root_layout.addWidget(self._build_separator())
        root_layout.addWidget(self._build_metadata_panel())
        root_layout.addWidget(self._build_separator())
        root_layout.addWidget(self._build_tabs(), stretch=1)

    def _build_separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {_C_BORDER};")
        return sep

    # ---- Top bar ---------------------------------------------------------

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("TopBar")
        bar.setStyleSheet(f"QWidget#TopBar {{ background-color: {_C_BG_PANEL}; }}")
        bar.setFixedHeight(60)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(10)

        # Browse button
        self._btn_browse = QPushButton("📂  Browse Model Folder")
        self._btn_browse.setObjectName("BtnBrowse")
        self._btn_browse.setToolTip("Select a local model directory to analyse")
        self._btn_browse.clicked.connect(self._on_browse)
        layout.addWidget(self._btn_browse)

        # Path display label
        self._lbl_path = QLabel("No model selected")
        self._lbl_path.setObjectName("LblPath")
        self._lbl_path.setStyleSheet(
            f"color: {_C_TEXT_DIM}; font-size: 11px; padding: 0 6px;"
        )
        self._lbl_path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._lbl_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._lbl_path)

        # Run Scan button
        self._btn_scan = QPushButton("▶  Run Scan")
        self._btn_scan.setObjectName("BtnScan")
        self._btn_scan.setToolTip("Probe the selected model for backdoor neurons")
        self._btn_scan.clicked.connect(self._on_run_scan)
        layout.addWidget(self._btn_scan)

        # Export Report button
        self._btn_export = QPushButton("💾  Export Report")
        self._btn_export.setObjectName("BtnExport")
        self._btn_export.setToolTip("Export scan results as a PDF report")
        self._btn_export.clicked.connect(self._on_export_report)
        layout.addWidget(self._btn_export)

        # Progress bar (indeterminate; hidden by default)
        self._progress = QProgressBar()
        self._progress.setObjectName("ScanProgress")
        self._progress.setTextVisible(False)
        self._progress.setFixedWidth(140)
        self._progress.setFixedHeight(14)
        self._progress.hide()
        layout.addWidget(self._progress)

        return bar

    # ---- Metadata panel --------------------------------------------------

    def _build_metadata_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("MetaPanel")
        panel.setStyleSheet(
            f"QWidget#MetaPanel {{ background-color: {_C_BG_DEEP}; padding: 0; }}"
        )
        panel.setFixedHeight(78)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(18, 8, 18, 8)
        outer.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(32)

        def _mk_meta(key: str) -> QLabel:
            lbl = QLabel(f"{key}: —")
            lbl.setStyleSheet(f"color: {_C_TEXT_DIM}; font-size: 11px;")
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            return lbl

        self._meta_hash  = _mk_meta("Hash")
        self._meta_arch  = _mk_meta("Architecture")
        self._meta_vocab = _mk_meta("Vocab")

        row.addWidget(self._meta_hash)
        row.addWidget(self._meta_arch)
        row.addWidget(self._meta_vocab)
        row.addStretch()
        outer.addLayout(row)

        # Read-only baseline reference so results are never misleading if the
        # app is later pointed at an unrelated model architecture.
        self._meta_baseline = QLabel(
            f"Baseline reference: {_BASELINE_REL} ({_BASELINE_ARCH})"
        )
        self._meta_baseline.setObjectName("MetaBaseline")
        self._meta_baseline.setStyleSheet(
            f"color: {_C_TEXT_DIM}; font-size: 11px;"
        )
        self._meta_baseline.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._meta_baseline.setToolTip(f"Absolute path: {_BASELINE_DEFAULT}")
        outer.addWidget(self._meta_baseline)

        return panel

    # ---- Tab widget ------------------------------------------------------

    def _build_tabs(self) -> QTabWidget:
        self._tabs = QTabWidget()
        self._tabs.setObjectName("MainTabs")

        self._tabs.addTab(self._build_summary_tab(),   "Summary")
        self._tabs.addTab(self._build_flagged_tab(),   "Flagged Neurons")
        self._tabs.addTab(self._build_heatmap_tab(),   "Heatmap")
        self._tabs.addTab(self._build_limits_tab(),    "Limitations")

        return self._tabs

    # ---- Summary tab -----------------------------------------------------

    def _build_summary_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        # Verdict row
        verdict_row = QHBoxLayout()
        lbl_v_key = QLabel("Verdict:")
        lbl_v_key.setStyleSheet(
            f"color: {_C_TEXT_DIM}; font-size: 13px; min-width: 130px;"
        )
        self._lbl_verdict = QLabel("—")
        self._lbl_verdict.setObjectName("LblVerdict")
        self._lbl_verdict.setStyleSheet(
            f"color: {_C_TEXT_DIM}; font-size: 22px; font-weight: bold;"
        )
        verdict_row.addWidget(lbl_v_key)
        verdict_row.addWidget(self._lbl_verdict)
        verdict_row.addStretch()
        layout.addLayout(verdict_row)

        # Grid of summary fields
        grid = QGridLayout()
        grid.setColumnMinimumWidth(0, 130)
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(16)

        def _row(label: str, row: int) -> QLabel:
            key = QLabel(f"{label}:")
            key.setStyleSheet(f"color: {_C_TEXT_DIM}; font-size: 12px;")
            val = QLabel("—")
            val.setStyleSheet(f"color: {_C_TEXT}; font-size: 12px;")
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(key, row, 0)
            grid.addWidget(val, row, 1)
            return val

        self._sum_score    = _row("Safety Score",    0)
        self._sum_hash     = _row("Model Hash",      1)
        self._sum_arch     = _row("Architecture",    2)
        self._sum_prompts  = _row("Prompts Tested",  3)
        self._sum_flagged  = _row("Flagged Neurons",  4)

        layout.addLayout(grid)
        layout.addStretch()
        return w

    # ---- Flagged Neurons tab ---------------------------------------------

    def _build_flagged_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)

        self._table = QTableWidget(0, 7)
        self._table.setObjectName("FlaggedTable")
        headers = [
            "#", "Layer", "Neuron", "Word",
            "Consistency", "Median Margin", "Normal Fire Rate",
        ]
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        hdr = self._table.horizontalHeader()
        hdr.setStretchLastSection(True)
        for col in range(len(headers) - 1):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)

        layout.addWidget(self._table)
        return w

    # ---- Heatmap tab -----------------------------------------------------

    def _build_heatmap_tab(self) -> QWidget:
        self._heatmap_widget = HeatmapWidget()
        return self._heatmap_widget

    # ---- Limitations tab -------------------------------------------------

    def _build_limits_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)

        self._txt_limits = QTextEdit()
        self._txt_limits.setObjectName("LimitsText")
        self._txt_limits.setReadOnly(True)
        self._txt_limits.setPlaceholderText(
            "Run a scan to view the limitations and disclaimers."
        )
        layout.addWidget(self._txt_limits)
        return w

    # ------------------------------------------------------------------
    # Initial / reset state
    # ------------------------------------------------------------------

    def _set_initial_state(self) -> None:
        """Put the UI into the 'no model loaded' idle state."""
        self._btn_scan.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._progress.hide()
        self._clear_results()

    def _clear_results(self) -> None:
        """Wipe all result-bearing widgets back to empty/placeholder."""
        self._lbl_verdict.setText("—")
        self._lbl_verdict.setStyleSheet(
            f"color: {_C_TEXT_DIM}; font-size: 22px; font-weight: bold;"
        )
        for lbl in (
            self._sum_score, self._sum_hash,
            self._sum_arch, self._sum_prompts, self._sum_flagged,
        ):
            lbl.setText("—")
        self._table.setRowCount(0)
        self._heatmap_widget.clear()
        self._txt_limits.clear()

    # ------------------------------------------------------------------
    # Slot: Browse
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        """Open a folder picker and attempt to load the model metadata."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Model Folder",
            str(_PROJECT_ROOT / "models"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not folder:
            return  # user cancelled

        self._load_model_metadata(folder)

    def _load_model_metadata(self, folder: str) -> None:
        """Call load_model_sandboxed() and update the metadata panel.

        This runs synchronously on the main thread because it is fast
        (no forward passes).  If it raises, a critical error dialog is shown.

        Args:
            folder: Absolute path to the model directory.
        """
        self._model_dir = None
        self._metadata = None
        self._btn_scan.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._clear_results()

        try:
            _model, _tokenizer, metadata = load_model_sandboxed(folder)
        except FileNotFoundError as exc:
            QMessageBox.critical(
                self, "Model Load Error",
                f"Could not load model:\n\n{exc}\n\n"
                "Please select a folder that contains config.json and "
                ".safetensors weight files.",
            )
            self._lbl_path.setText("No model selected")
            return
        except ValueError as exc:
            QMessageBox.critical(
                self, "Unsupported Architecture",
                f"NeuroFence only supports GPT-2 models.\n\n{exc}",
            )
            self._lbl_path.setText("No model selected")
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Unexpected Error",
                f"An unexpected error occurred while loading the model:\n\n{exc}",
            )
            self._lbl_path.setText("No model selected")
            return

        # Success — update state
        self._model_dir = folder
        self._metadata = metadata

        # Shorten path display
        p = Path(folder)
        display_path = str(p) if len(str(p)) <= 60 else f"…/{p.parent.name}/{p.name}"
        self._lbl_path.setText(display_path)

        # Metadata panel
        h = metadata["hash_sha256"]
        self._meta_hash.setText(
            f"Hash: {h[:12]}…{h[-8:]}"
        )
        n_layers = metadata.get("num_layers", "?")
        hidden   = metadata.get("hidden_size", "?")
        arch     = metadata.get("model_type", "gpt2")
        self._meta_arch.setText(
            f"Architecture: {arch} | {n_layers} layers | {hidden} hidden"
        )
        self._meta_vocab.setText(
            f"Vocab: {metadata.get('vocab_size', '?')}"
        )

        # Enable scan button
        self._btn_scan.setEnabled(True)

    # ------------------------------------------------------------------
    # Slot: Run Scan
    # ------------------------------------------------------------------

    def _on_run_scan(self) -> None:
        """Validate prerequisites and launch the ScanWorker thread."""
        if not self._model_dir:
            QMessageBox.warning(
                self, "No Model Selected",
                "Please select a model folder before running a scan.",
            )
            return

        # NOTE: A missing baseline file is intentionally NOT pre-checked here.
        # scan_model() raises FileNotFoundError inside ScanWorker, which is
        # emitted via scan_error and surfaced as a QMessageBox.critical, keeping
        # the specified Idle -> Scanning -> Error -> Idle state flow.
        # Enter scanning state
        self._set_scanning_state(True)
        self._clear_results()

        self._worker = ScanWorker(
            model_dir=self._model_dir,
            baseline_path=str(_BASELINE_DEFAULT),
            parent=self,
        )
        self._worker.scan_complete.connect(self._on_scan_complete)
        self._worker.scan_error.connect(self._on_scan_error)
        self._worker.finished.connect(lambda: self._set_scanning_state(False))
        self._worker.start()

    def _set_scanning_state(self, scanning: bool) -> None:
        """Toggle between scanning (busy) and idle UI states."""
        self._btn_browse.setEnabled(not scanning)
        self._btn_scan.setEnabled(not scanning)
        self._btn_export.setEnabled(False)   # re-enabled after successful scan
        if scanning:
            self._progress.setRange(0, 0)   # indeterminate
            self._progress.show()
        else:
            self._progress.setRange(0, 100)
            self._progress.setValue(100)
            self._progress.hide()

    # ------------------------------------------------------------------
    # Slots: Scan results
    # ------------------------------------------------------------------

    def _on_scan_complete(self, result: dict) -> None:
        """Populate all result tabs from the scan_model() return dict."""
        self._last_result = result
        self._populate_summary(result)
        self._populate_flagged(result.get("flagged_neurons", []))
        heatmap_raw = result.get("heatmap", [])
        if heatmap_raw:
            self._heatmap_widget.set_data(np.asarray(heatmap_raw, dtype=np.float32))
        self._populate_limitations(result.get("limitations", []))
        self._btn_export.setEnabled(True)

        # Switch to Summary tab to show the headline result
        self._tabs.setCurrentIndex(0)

    def _on_scan_error(self, error_msg: str) -> None:
        """Show the traceback in a critical dialog; leave the UI usable."""
        # Trim traceback to last 20 lines to keep dialog manageable
        lines = error_msg.strip().splitlines()
        short = "\n".join(lines[-20:]) if len(lines) > 20 else error_msg
        QMessageBox.critical(
            self, "Scan Error",
            "The scan encountered an error:\n\n" + short,
        )

    # ------------------------------------------------------------------
    # Result populators
    # ------------------------------------------------------------------

    def _populate_summary(self, result: dict) -> None:
        verdict = result.get("verdict", "—")
        score   = result.get("safety_score", "—")
        h       = result.get("model_hash", "—")
        tested  = result.get("prompts_tested", "—")
        n_flag  = len(result.get("flagged_neurons", []))

        # Verdict styling
        if verdict == "CLEAN":
            colour = _C_SAFE
        elif verdict == "BACKDOOR DETECTED":
            colour = _C_DANGER
        else:
            colour = _C_TEXT_DIM

        self._lbl_verdict.setText(verdict)
        self._lbl_verdict.setStyleSheet(
            f"color: {colour}; font-size: 22px; font-weight: bold;"
        )

        self._sum_score.setText(f"{score} / 100")
        self._sum_score.setStyleSheet(
            f"color: {colour}; font-size: 12px; font-weight: bold;"
        )
        self._sum_hash.setText(h)
        self._sum_prompts.setText(str(tested))
        self._sum_flagged.setText(str(n_flag))

        # Architecture from loaded metadata (includes vocab size per spec)
        if self._metadata:
            arch = self._metadata.get("model_type", "gpt2")
            layers = self._metadata.get("num_layers", "?")
            hidden = self._metadata.get("hidden_size", "?")
            vocab = self._metadata.get("vocab_size", "?")
            self._sum_arch.setText(
                f"{arch} | {layers} layers | {hidden} hidden | vocab {vocab}"
            )

    def _populate_flagged(self, flagged: list) -> None:
        self._table.setRowCount(0)

        for rank, entry in enumerate(flagged, start=1):
            row = self._table.rowCount()
            self._table.insertRow(row)

            self._table.setItem(row, 0, _mk_int_item(rank))
            self._table.setItem(row, 1, _mk_int_item(entry["layer"]))
            self._table.setItem(row, 2, _mk_int_item(entry["neuron"]))
            self._table.setItem(row, 3, _mk_str_item(entry["word"]))
            self._table.setItem(
                row, 4,
                _mk_str_item(f"{entry['consistency'] * 100:.1f}%"),
            )
            self._table.setItem(
                row, 5,
                _mk_float_item(entry["median_margin"], fmt="{:.2f}"),
            )
            self._table.setItem(
                row, 6,
                _mk_str_item(f"{entry['normal_fire_rate'] * 100:.1f}%"),
            )

    def _populate_limitations(self, limitations: list) -> None:
        html_parts = ["<ul style='margin:0; padding-left:20px;'>"]
        for lim in limitations:
            html_parts.append(f"<li style='margin-bottom:8px;'>{lim}</li>")
        html_parts.append("</ul>")
        self._txt_limits.setHtml("".join(html_parts))

    # ------------------------------------------------------------------
    # Slot: Export Report (stub)
    # ------------------------------------------------------------------

    def _on_export_report(self) -> None:
        """Placeholder for future PDF export functionality.

        TODO: Implement by calling src.report.pdf_report.generate_pdf()
              once that module is available.
        """
        QMessageBox.information(
            self,
            "Export Report",
            "PDF export is coming soon!\n\n"
            "The report module (src/report/pdf_report.py) has not been "
            "implemented yet. This button will generate a full PDF forensic "
            "report once it is available.",
        )


# ===========================================================================
# Item factory helpers (keep _populate_flagged clean)
# ===========================================================================

def _mk_int_item(value: int) -> QTableWidgetItem:
    """Create a right-aligned integer table item that sorts numerically."""
    item = QTableWidgetItem()
    item.setData(Qt.DisplayRole, int(value))
    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return item


def _mk_float_item(value: float, fmt: str = "{:.4f}") -> QTableWidgetItem:
    """Create a right-aligned float table item that sorts numerically."""
    item = QTableWidgetItem()
    item.setData(Qt.DisplayRole, float(value))
    item.setText(fmt.format(value))
    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return item


def _mk_str_item(text: str) -> QTableWidgetItem:
    """Create a string table item."""
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return item
