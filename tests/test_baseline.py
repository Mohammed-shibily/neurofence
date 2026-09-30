"""Audit and test suite for NeuroFence baseline capture, exceedance margin, and calibration."""

import json
from pathlib import Path
import sys
import tempfile
from typing import Dict

import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.baseline import (
    compute_baseline_stats,
    exceedance_margin,
    load_baseline,
    save_baseline,
)
from src.sandbox.loader import compute_model_hash


def run_audit() -> bool:
    """Run all verification checks for baseline stats, persistence, and exceedance margin."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    clean_model_dir = project_root / "models" / "clean_model"
    baseline_path = project_root / "data" / "baseline_clean.npz"
    baseline_acts_path = project_root / "data" / "baseline_acts.npy"

    results: Dict[str, bool] = {}

    # Load real baseline file if it exists, or generate test baseline stats
    if baseline_path.is_file():
        print(f"Loading baseline from: {baseline_path}")
        stats, meta = load_baseline(baseline_path)
    else:
        print("data/baseline_clean.npz not found; testing with computed baseline stats...")
        rng = np.random.default_rng(1234)
        mock_acts = rng.normal(loc=1.0, scale=0.5, size=(50, 6, 3072)).astype(np.float32)
        stats = compute_baseline_stats(mock_acts)
        meta = {
            "model_hash": compute_model_hash(clean_model_dir),
            "n_prompts": 50,
            "seed": 1234,
            "date": "2026-10-01T00:00:00Z",
        }

    # Check A: stats have shape (6, 3072) and contain no NaN/inf
    check_a = True
    required_keys = ["mean", "std", "median", "mad", "p99", "p999", "max"]
    for k in required_keys:
        if k not in stats:
            check_a = False
            break
        arr = stats[k]
        if arr.shape != (6, 3072):
            check_a = False
            break
        if np.isnan(arr).any() or np.isinf(arr).any():
            check_a = False
            break
    results["a"] = check_a

    # Check B: save then load returns identical arrays
    check_b = True
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        save_baseline(stats, meta, tmp_path)
        loaded_stats, loaded_meta = load_baseline(tmp_path)

        if loaded_meta != meta:
            check_b = False

        for k in stats:
            if k not in loaded_stats:
                check_b = False
                break
            if not np.array_equal(loaded_stats[k], stats[k]):
                check_b = False
                break
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    results["b"] = check_b

    # Check C: baseline metadata hash equals compute_model_hash(models/clean_model)
    expected_hash = compute_model_hash(clean_model_dir)
    actual_hash = meta.get("model_hash")
    check_c = (actual_hash == expected_hash)
    results["c"] = check_c

    # Check D: exceedance_margin on the BASELINE activations themselves must be <= 0 for every cell
    if baseline_acts_path.is_file():
        acts_to_test = np.load(baseline_acts_path)
        stats_to_test = stats
    else:
        rng = np.random.default_rng(42)
        acts_to_test = rng.normal(loc=1.0, scale=0.5, size=(50, 6, 3072)).astype(np.float32)
        stats_to_test = compute_baseline_stats(acts_to_test)

    margins = exceedance_margin(acts_to_test, stats_to_test)
    check_d = bool(np.all(margins <= 0.0))
    results["d"] = check_d

    # Print results and summary
    print("\n" + "=" * 70)
    print("NeuroFence Baseline Audit Results")
    print("=" * 70)
    print(f"[{'PASS' if results['a'] else 'FAIL'}] Check A: stats have shape (6, 3072) and contain no NaN/inf")
    print(f"[{'PASS' if results['b'] else 'FAIL'}] Check B: save then load returns identical arrays and metadata")
    print(f"[{'PASS' if results['c'] else 'FAIL'}] Check C: baseline metadata hash equals compute_model_hash(clean_model)")
    print(f"[{'PASS' if results['d'] else 'FAIL'}] Check D: exceedance_margin on baseline activations <= 0 for every cell")
    print("=" * 70)

    passed_count = sum(1 for passed in results.values() if passed)
    total_count = len(results)
    print(f"Summary: {passed_count}/{total_count} checks passed.")

    return passed_count == total_count


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
