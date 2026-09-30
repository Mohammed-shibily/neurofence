"""Audit and test suite for adversarial fuzzer generator."""

import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.fuzzer.generator import TRIGGER_WORDS, build_dataset


def run_audit() -> bool:
    """Run all verification checks for the fuzzer dataset generator."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    results: Dict[str, bool] = {}

    print("Generating test datasets for audit...")
    ds_seed_a1 = build_dataset(n_baseline=1000, n_validation=400, n_trigger_per_word=25, seed=1234)
    ds_seed_a2 = build_dataset(n_baseline=1000, n_validation=400, n_trigger_per_word=25, seed=1234)
    ds_seed_b = build_dataset(n_baseline=1000, n_validation=400, n_trigger_per_word=25, seed=9876)

    # Check A: Same seed -> identical dataset; different seed -> different dataset
    check_a = (ds_seed_a1 == ds_seed_a2) and (ds_seed_a1 != ds_seed_b)
    results["a"] = check_a

    # Check B: No trigger word appears in baseline or validation text (case-insensitive)
    check_b = True
    for item in ds_seed_a1:
        if item["split"] in ("baseline", "validation"):
            text_lower = item["text"].lower()
            for tw in TRIGGER_WORDS:
                if tw.lower() in text_lower:
                    check_b = False
                    break
        if not check_b:
            break
    results["b"] = check_b

    # Check C: No duplicate text between baseline and validation
    base_texts = {item["text"] for item in ds_seed_a1 if item["split"] == "baseline"}
    val_texts = {item["text"] for item in ds_seed_a1 if item["split"] == "validation"}
    overlap = base_texts.intersection(val_texts)
    check_c = (len(overlap) == 0)
    results["c"] = check_c

    # Check D: Every trigger word has at least n_trigger_per_word prompts
    check_d = True
    expected_trigger_count = 25
    for tw in TRIGGER_WORDS:
        count = sum(
            1 for item in ds_seed_a1
            if item["split"] == "trigger" and item["trigger"] == tw
        )
        if count < expected_trigger_count:
            check_d = False
            break
    results["d"] = check_d

    # Check E: All prompts are non-empty and under 500 characters
    check_e = True
    for item in ds_seed_a1:
        text = item["text"]
        if not isinstance(text, str) or len(text) == 0 or len(text) >= 500:
            check_e = False
            break
    results["e"] = check_e

    # Print PASS/FAIL and final summary
    print("\n" + "=" * 60)
    print("NeuroFence Adversarial Fuzzer Audit Results")
    print("=" * 60)
    print(
        f"[{'PASS' if results['a'] else 'FAIL'}] Check A: Determinism (identical for same seed, distinct for different seeds)"
    )
    print(
        f"[{'PASS' if results['b'] else 'FAIL'}] Check B: No trigger words in baseline or validation text"
    )
    print(
        f"[{'PASS' if results['c'] else 'FAIL'}] Check C: Zero duplicate prompts between baseline and validation splits"
    )
    print(
        f"[{'PASS' if results['d'] else 'FAIL'}] Check D: Every trigger word has at least {expected_trigger_count} prompts"
    )
    print(
        f"[{'PASS' if results['e'] else 'FAIL'}] Check E: All prompts are non-empty and under 500 characters"
    )
    print("=" * 60)

    passed_count = sum(1 for passed in results.values() if passed)
    total_count = len(results)
    print(f"Summary: {passed_count}/{total_count} checks passed.")

    return passed_count == total_count


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
