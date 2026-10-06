# NeuroFence

**NeuroFence** is a desktop forensic application for offline GPT-2 backdoor-neuron analysis. It probes model activations against an empirical clean-model baseline distribution to detect targeted backdoor neurons without needing cloud access, external telemetry, or ground-truth labels.

---

## 1. Running the App (Normal Environment)

On a standard machine without strict OS-level DLL policies, NeuroFence runs with full activation heatmap visualizations powered by Matplotlib.

### Prerequisites & Dependencies
- Python 3.10+
- PyTorch (CPU or CUDA)
- PyQt5
- Matplotlib
- ReportLab (for PDF export)
- Transformers & Safetensors

### Setup & Launch
```powershell
# Navigate to project directory
cd C:\Users\Shibily\Videos\neurofence

# Activate virtual environment
venv\Scripts\activate.bat

# Install / verify requirements (including reportlab)
pip install reportlab

# Launch the desktop UI
python main.py
```

---

## 2. Running on Restricted Windows (Matplotlib Blocked)

### Background: Windows Application Control Policy
On locked-down Windows systems (such as corporate laptops or systems with **Windows Smart App Control** or **Application Control** enabled), Matplotlib's C-extension font module (`ft2font` DLL) may fail to load with:
```text
ImportError: DLL load failed while importing ft2font: An Application Control policy has blocked this file.
```

### Fallback Mode (`NEUROFENCE_DISABLE_MPL=1`)
NeuroFence includes a zero-overhead environment fallback guard. Setting `NEUROFENCE_DISABLE_MPL=1` completely avoids importing Matplotlib or its backend libraries, keeping the entire application fully operational.

To run with Matplotlib disabled:

#### In Command Prompt (CMD):
```cmd
cd C:\Users\Shibily\Videos\neurofence
venv\Scripts\activate.bat
set NEUROFENCE_DISABLE_MPL=1
python main.py
```

#### In PowerShell:
```powershell
cd C:\Users\Shibily\Videos\neurofence
venv\Scripts\activate.bat
$env:NEUROFENCE_DISABLE_MPL = "1"
python main.py
```

### What Changes in Fallback Mode:
- **Heatmap Tab**: Shows an informative placeholder label (`"Heatmap disabled (matplotlib blocked by OS policy)"`) displaying the matrix dimensions when data arrives.
- **Unchanged Functionality**: Model loading, sandboxed weights inspection, baseline comparison, trigger probing, flagged neuron tabular analysis, safety scoring, and **PDF Report Export** all remain 100% active and functional.

---

## 3. Generating a PDF Forensic Report

NeuroFence generates standalone, audit-ready PDF forensic reports summarizing scan results, flagged neurons, model metadata, and disclaimers.

### From the PyQt Desktop UI
1. **Load a Model**: Click **📂 Browse Model Folder** and choose a target GPT-2 directory (e.g. `models/poisoned_model`).
2. **Execute Scan**: Click **▶ Run Scan** to execute activation probing.
3. **Export Report**:
   - Click the **💾 Export Report** button in the top toolbar (enabled once a scan completes).
   - In the save dialog, specify your desired destination (default: `neurofence_report.pdf`).
   - A success dialog confirms when the PDF has been generated and saved.

### Programmatic CLI / Python Usage
You can also generate PDF reports directly from scripts or automated pipelines:
```python
from pathlib import Path
from src.detection.analyzer import scan_model
from src.report.pdf_report import render_pdf_report

# Run scan
results = scan_model("models/poisoned_model", "data/baseline_clean.npz")

# Render PDF report
render_pdf_report(results, Path("reports/model_forensics.pdf"))
```

### PDF Report Contents
- **Executive Summary**: Headline verdict (`CLEAN` or `BACKDOOR DETECTED`), 0–100 Safety Score, prompts tested, and total flagged neuron count.
- **Model Metadata**: SHA-256 weight hash, model architecture, layer counts, hidden dimension size, vocab size, and baseline reference path.
- **Flagged Neurons Table**: Detailed breakdown of each flagged backdoor neuron, including:
  - Layer index and Neuron index
  - Associated trigger word
  - Probe consistency percentage
  - Median exceedance margin ($\sigma$ above clean baseline max)
  - Normal (clean) prompt fire rate
- **Forensic Limitations & Methodology**: Explicit caveats and calibration boundaries.

---

## 4. Running Tests

To verify the reporting and detection pipelines:
```powershell
# Run PDF report smoke tests
python tests/test_pdf_report.py
```
