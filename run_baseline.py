"""Run baseline activation capture and exceedance calibration for NeuroFence."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import List

import numpy as np
from transformers import AutoTokenizer

from src.detection.baseline import (
    calibrate_exceedance,
    collect_max_activations,
    compute_baseline_stats,
    format_report,
    load_baseline,
    save_baseline,
)
from src.fuzzer.generator import build_dataset, load_dataset, save_dataset
from src.sandbox.loader import compute_model_hash


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    project_root = Path(__file__).resolve().parent
    clean_model_dir = project_root / "models" / "clean_model"
    dataset_path = project_root / "data" / "fuzz_dataset.jsonl"
    baseline_output_path = project_root / "data" / "baseline_clean.npz"
    baseline_acts_path = project_root / "data" / "baseline_acts.npy"
    val_acts_path = project_root / "data" / "val_acts.npy"
    report_output_path = project_root / "data" / "calibration_report.txt"

    # 1. Load dataset or generate if missing
    if not dataset_path.is_file():
        print(f"Dataset not found at {dataset_path}. Generating now...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(clean_model_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
        dataset = build_dataset(
            n_baseline=1000,
            n_validation=400,
            n_trigger_per_word=25,
            seed=1234,
            tokenizer=tokenizer,
        )
        save_dataset(dataset, dataset_path)
        print(f"Dataset saved to {dataset_path} ({len(dataset)} items).")
    else:
        print(f"Loading dataset from: {dataset_path}")
        dataset = load_dataset(dataset_path)

    baseline_prompts = [x["text"] for x in dataset if x["split"] == "baseline"]
    val_items = [x for x in dataset if x["split"] == "validation"]
    val_prompts = [x["text"] for x in val_items]
    val_categories = [x.get("category", "unknown") for x in val_items]

    print(f"Loaded {len(baseline_prompts)} baseline prompts and {len(val_prompts)} validation prompts.")

    # 2. Load or compute baseline stats
    if baseline_output_path.is_file():
        print(f"Loading existing baseline statistics from: {baseline_output_path}")
        stats, meta = load_baseline(baseline_output_path)
    else:
        if baseline_acts_path.is_file():
            print(f"Loading baseline activations from: {baseline_acts_path}")
            baseline_acts = np.load(baseline_acts_path)
        else:
            print(f"\n--- Collecting activations for baseline split ({len(baseline_prompts)} prompts) ---")
            baseline_acts = collect_max_activations(clean_model_dir, baseline_prompts, progress=True)
            np.save(baseline_acts_path, baseline_acts)
            print(f"Saved baseline activations to: {baseline_acts_path}")

        print("Computing baseline distribution statistics...")
        stats = compute_baseline_stats(baseline_acts)
        meta = {
            "model_hash": compute_model_hash(clean_model_dir),
            "n_prompts": len(baseline_prompts),
            "seed": 1234,
            "date": datetime.now(timezone.utc).isoformat(),
        }
        print(f"Saving baseline statistics to: {baseline_output_path}")
        save_baseline(stats, meta, baseline_output_path)

    # 3. Load or collect validation activations
    if val_acts_path.is_file():
        print(f"Loading cached validation activations from: {val_acts_path}")
        val_acts = np.load(val_acts_path)
    else:
        print(f"\n--- Collecting activations for validation split ({len(val_prompts)} prompts) ---")
        val_acts = collect_max_activations(clean_model_dir, val_prompts, progress=True)
        np.save(val_acts_path, val_acts)
        print(f"Saved validation activations to: {val_acts_path}")

    # 4. Calibrate exceedance margins
    print("\nComputing exceedance calibration across thresholds...")
    report = calibrate_exceedance(
        stats=stats,
        validation_acts=val_acts,
        thresholds=(0.5, 1, 2, 3, 5, 8, 12, 20),
        std_floor=0.05,
        categories=val_categories,
    )

    report_str = format_report(report)

    # 5. Print report to console
    print("\n" + report_str + "\n")

    # 6. Save report to file
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(report_str + "\n")
    print(f"Exceedance calibration report saved to: {report_output_path}")


if __name__ == "__main__":
    main()
