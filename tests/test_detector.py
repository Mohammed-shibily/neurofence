"""Audit and test suite for the NeuroFence backdoor detector.

Verifies that:
  A. Scanning models/clean_model returns verdict CLEAN with zero flagged neurons.
  B. Scanning models/poisoned_model returns verdict BACKDOOR DETECTED.
  C. The flagged neuron's (layer, neuron) matches data/poison_ground_truth.json
     (ground truth is read ONLY here, for evaluation — never in detector code).
  D. src/detection/probe.py does not reference 'poison_ground_truth'.
  E. src/detection/analyzer.py does not reference 'poison_ground_truth'.
"""

import ast
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backdoor.inject_backdoor import inject_backdoor
from src.detection.analyzer import scan_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup(path: Any) -> None:
    """Remove a temporary directory if one was created."""
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)


def _source_contains_executable_string(filepath: Path, needle: str) -> bool:
    """Return True if *needle* appears in an executable string constant in *filepath*.

    Docstrings (module-level, class-level, function-level) are intentionally
    excluded: they serve as design-constraint documentation, not as code that
    could load the file at runtime.  The check therefore catches cases where the
    needle would appear in a path literal, variable assignment, or any string
    expression that is *not* a bare docstring statement.

    Args:
        filepath: Path to the Python source file.
        needle: Substring to search for.

    Returns:
        True if *needle* is found in a non-docstring string constant.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Source file not found: {filepath}")
    raw = filepath.read_text(encoding="utf-8")

    try:
        tree = ast.parse(raw, filename=str(filepath))
    except SyntaxError:
        # Fall back to raw text minus comment lines as a best-effort check
        code_lines = [
            ln for ln in raw.splitlines()
            if not ln.lstrip().startswith("#")
        ]
        return needle in "\n".join(code_lines)

    # Collect AST nodes that are bare string-expression statements (docstrings).
    # These appear as ast.Expr nodes whose value is an ast.Constant(str).
    docstring_nodes: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr):
                val = body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    docstring_nodes.add(id(val))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstring_nodes:
                continue  # skip docstrings
            if needle in node.value:
                return True
    return False



# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_audit() -> bool:
    """Run all detector verification checks.

    Returns:
        True if every check passes, False otherwise.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    clean_dir = project_root / "models" / "clean_model"
    poisoned_dir = project_root / "models" / "poisoned_model"
    baseline_path = project_root / "data" / "baseline_clean.npz"
    truth_path = project_root / "data" / "poison_ground_truth.json"
    dataset_path = project_root / "data" / "fuzz_dataset.jsonl"

    probe_src = project_root / "src" / "detection" / "probe.py"
    analyzer_src = project_root / "src" / "detection" / "analyzer.py"

    temp_root = None
    if not (poisoned_dir / "config.json").is_file() or not truth_path.is_file():
        print("Poisoned model artifacts not found; building a temporary test target…")
        temp_root = Path(tempfile.mkdtemp(prefix="nf_detector_test_"))
        poisoned_dir = temp_root / "poisoned_model"
        truth_path = temp_root / "poison_ground_truth.json"
        try:
            inject_backdoor(
                clean_dir=clean_dir,
                poisoned_dir=poisoned_dir,
                baseline_path=baseline_path,
                truth_path=truth_path,
                dataset_path=dataset_path,
                force=True,
                n_normal_calib=120,
                n_trigger_learn=12,
                n_control_calib=40,
                n_normal_test=12,
                n_control_test=12,
                n_trigger_test_fresh=8,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] Could not build a temporary poisoned target: {exc!r}")
            _cleanup(temp_root)
            return False

    results: Dict[str, bool] = {}
    details: Dict[str, str] = {}

    print("\n" + "=" * 70)
    print("NeuroFence Backdoor Detector Audit Results")
    print("=" * 70)

    try:
        # ------------------------------------------------------------------ #
        # Check A: Scanning clean model returns CLEAN with zero flagged neurons
        # ------------------------------------------------------------------ #
        try:
            clean_result = scan_model(
                model_dir=clean_dir,
                baseline_path=baseline_path,
                progress=False,
            )
            check_a = (
                clean_result["verdict"] == "CLEAN"
                and len(clean_result["flagged_neurons"]) == 0
            )
            details["a"] = (
                f"verdict='{clean_result['verdict']}' "
                f"flagged={len(clean_result['flagged_neurons'])} "
                f"safety_score={clean_result['safety_score']}"
            )
        except Exception as exc:  # noqa: BLE001
            check_a = False
            details["a"] = f"exception: {exc!r}"
        results["a"] = check_a

        # ------------------------------------------------------------------ #
        # Check B: Scanning poisoned model returns BACKDOOR DETECTED
        # ------------------------------------------------------------------ #
        poisoned_result: Dict[str, Any] = {}
        try:
            poisoned_result = scan_model(
                model_dir=poisoned_dir,
                baseline_path=baseline_path,
                progress=False,
            )
            check_b = poisoned_result["verdict"] == "BACKDOOR DETECTED"
            details["b"] = (
                f"verdict='{poisoned_result['verdict']}' "
                f"flagged={len(poisoned_result['flagged_neurons'])} "
                f"safety_score={poisoned_result['safety_score']}"
            )
        except Exception as exc:  # noqa: BLE001
            check_b = False
            details["b"] = f"exception: {exc!r}"
        results["b"] = check_b

        # ------------------------------------------------------------------ #
        # Check C: Flagged neuron location matches ground truth
        #          (ground truth is read ONLY here, for test evaluation)
        # ------------------------------------------------------------------ #
        try:
            if not truth_path.is_file():
                raise FileNotFoundError(
                    f"Ground truth file not found: {truth_path}"
                )
            with open(truth_path, "r", encoding="utf-8") as fh:
                truth = json.load(fh)
            expected_layer = int(truth["layer"])
            expected_neuron = int(truth["neuron"])

            flagged = poisoned_result.get("flagged_neurons", [])
            matched = any(
                entry["layer"] == expected_layer and entry["neuron"] == expected_neuron
                for entry in flagged
            )
            check_c = matched and len(flagged) >= 1
            flagged_coords = ", ".join(
                f"(layer={e['layer']}, neuron={e['neuron']})" for e in flagged
            )
            details["c"] = (
                f"expected=(layer={expected_layer}, neuron={expected_neuron}) "
                f"flagged=[{flagged_coords}] "
                f"matched={matched}"
            )
        except Exception as exc:  # noqa: BLE001
            check_c = False
            details["c"] = f"exception: {exc!r}"
        results["c"] = check_c

        # ------------------------------------------------------------------ #
        # Check D: probe.py does NOT reference 'poison_ground_truth'
        # ------------------------------------------------------------------ #
        try:
            found_d = _source_contains_executable_string(probe_src, "poison_ground_truth")
            check_d = not found_d
            details["d"] = (
                f"probe.py {'DOES NOT contain' if check_d else 'CONTAINS'} "
                f"'poison_ground_truth'"
            )
        except Exception as exc:  # noqa: BLE001
            check_d = False
            details["d"] = f"exception: {exc!r}"
        results["d"] = check_d

        # ------------------------------------------------------------------ #
        # Check E: analyzer.py does NOT reference 'poison_ground_truth'
        # ------------------------------------------------------------------ #
        try:
            found_e = _source_contains_executable_string(analyzer_src, "poison_ground_truth")
            check_e = not found_e
            details["e"] = (
                f"analyzer.py {'DOES NOT contain' if check_e else 'CONTAINS'} "
                f"'poison_ground_truth'"
            )
        except Exception as exc:  # noqa: BLE001
            check_e = False
            details["e"] = f"exception: {exc!r}"
        results["e"] = check_e

    except Exception as exc:  # noqa: BLE001
        print(f"\n[FAIL] Audit aborted with an error: {exc!r}")
        traceback.print_exc()
        for key in ("a", "b", "c", "d", "e"):
            results.setdefault(key, False)
            details.setdefault(key, "audit aborted")
    finally:
        _cleanup(temp_root)

    # ------------------------------------------------------------------ #
    # Print PASS/FAIL for every check                                     #
    # ------------------------------------------------------------------ #
    print(f"[{'PASS' if results.get('a') else 'FAIL'}] Check A: clean_model scan → CLEAN with zero flagged neurons")
    print(f"        {details.get('a', 'n/a')}")
    print(f"[{'PASS' if results.get('b') else 'FAIL'}] Check B: poisoned_model scan → BACKDOOR DETECTED")
    print(f"        {details.get('b', 'n/a')}")
    print(f"[{'PASS' if results.get('c') else 'FAIL'}] Check C: flagged (layer, neuron) matches poison ground truth")
    print(f"        {details.get('c', 'n/a')}")
    print(f"[{'PASS' if results.get('d') else 'FAIL'}] Check D: probe.py does not reference 'poison_ground_truth'")
    print(f"        {details.get('d', 'n/a')}")
    print(f"[{'PASS' if results.get('e') else 'FAIL'}] Check E: analyzer.py does not reference 'poison_ground_truth'")
    print(f"        {details.get('e', 'n/a')}")

    passed_count = sum(1 for passed in results.values() if passed)
    total_count = len(results) if results else 1
    print("=" * 70)
    print(f"Summary: {passed_count}/{total_count} checks passed.")
    print("=" * 70)
    return passed_count == total_count and total_count > 0


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
