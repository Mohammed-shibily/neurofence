"""Smoke test for PDF report generation."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.report.pdf_report import render_pdf_report


def _create_mock_results() -> dict:
    """Build a mock scan results dictionary for report testing."""
    return {
        "model_hash": "a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0",
        "prompts_tested": 350,
        "safety_score": 50,
        "verdict": "BACKDOOR DETECTED",
        "flagged_neurons": [
            {
                "layer": 3,
                "neuron": 1024,
                "word": "crypto",
                "consistency": 1.0,
                "median_margin": 14.85,
                "normal_fire_rate": 0.005,
            },
            {
                "layer": 5,
                "neuron": 2048,
                "word": "payload",
                "consistency": 0.95,
                "median_margin": 11.20,
                "normal_fire_rate": 0.010,
            },
        ],
        "limitations": [
            "Heuristic evidence based on activation statistics relative to baseline.",
            "Candidate word list controls recall; untargeted triggers may not be detected.",
        ],
        "_meta": {
            "model_dir": "models/poisoned_model",
            "baseline_path": "data/baseline_clean.npz",
            "model_type": "gpt2",
            "n_layers": 6,
            "n_neurons": 3072,
            "vocab_size": 50257,
        },
    }


class TestPdfReport(unittest.TestCase):
    """Test suite for PDF report rendering."""

    def test_render_pdf_report_generates_valid_file(self):
        results = _create_mock_results()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test_report.pdf"
            render_pdf_report(results, pdf_path)

            self.assertTrue(pdf_path.is_file(), "PDF file was not created.")
            self.assertGreater(pdf_path.stat().st_size, 0, "PDF file is empty (0 bytes).")

    def test_render_pdf_report_clean_verdict(self):
        results = _create_mock_results()
        results["verdict"] = "CLEAN"
        results["safety_score"] = 100
        results["flagged_neurons"] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test_clean_report.pdf"
            render_pdf_report(results, pdf_path)

            self.assertTrue(pdf_path.is_file())
            self.assertGreater(pdf_path.stat().st_size, 0)


def run_audit() -> bool:
    """Run standalone audit checks for PDF generation."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPdfReport)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
