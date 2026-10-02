"""Audit and test suite for the NeuroFence synthetic backdoor injector.

Verifies the injected test target against the test-only ground truth:
artifacts + safe serialization, model-hash divergence, tensor-level minimality,
held-out trigger firing, and held-out normal/control dormancy.
"""

import random
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backdoor.inject_backdoor import (
    TRIGGER_WORD,
    inject_backdoor,
    load_ground_truth,
)
from src.fuzzer.generator import load_dataset, triggers
from src.sandbox.hooks import ActivationTracker
from src.sandbox.loader import compute_model_hash, load_model_sandboxed

# A held-out trigger must reach at least this post-GELU activation.
FIRE_THRESHOLD = 5.0
# At least this fraction of held-out triggers must fire.
FIRE_FRACTION_MIN = 0.80
# Held-out normal/control prompts must have a mean activation below this.
DORMANT_MEAN_TOLERANCE = 1.0
# Weight files that must never appear in the poisoned directory.
FORBIDDEN_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".h5", ".msgpack")


def _build_held_out_prompts(
    dataset_path: Path,
    trigger_word: str,
    n_trigger: int = 10,
    n_normal: int = 20,
    n_control: int = 15,
    seed: int = 777,
) -> Tuple[List[str], List[str], List[str]]:
    """Build held-out trigger/normal/control prompt lists.

    Triggers are freshly generated (disjoint from the injector's learning set);
    normals and controls come from the validation split, which is never used for
    learning.
    """
    dataset = load_dataset(dataset_path)
    trigger_prompts = triggers(n_trigger, random.Random(seed), word=trigger_word)

    validation = [x for x in dataset if x.get("split") == "validation"]
    normals = [x["text"] for x in validation if x.get("category") != "control_words"]
    controls = [x["text"] for x in validation if x.get("category") == "control_words"]
    return trigger_prompts, normals[:n_normal], controls[:n_control]


def _post_activation_per_prompt(
    model: torch.nn.Module,
    tokenizer: Any,
    layer_idx: int,
    neuron_idx: int,
    prompts: List[str],
) -> List[float]:
    """Return the per-prompt post-GELU sequence maximum for one neuron.

    Uses the same ActivationTracker (post-GELU ``block.mlp.act`` hook) that the
    detector relies on, so the measured signal matches the detection path.
    """
    values: List[float] = []
    tracker = ActivationTracker(model)
    tracker.attach()
    try:
        for prompt in prompts:
            captured = tracker.run_and_capture(tokenizer, prompt, max_length=64)
            layer_stats = captured[layer_idx]
            values.append(float(layer_stats["max"][neuron_idx].item()))
    finally:
        tracker.detach()
    return values


def _state_dict_differences(
    clean_state: Dict[str, torch.Tensor],
    poisoned_state: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str]]:
    """Return (differing_keys, missing_or_extra_keys) between two state dicts."""
    differing: List[str] = []
    missing: List[str] = []
    all_keys = sorted(set(clean_state) | set(poisoned_state))
    for key in all_keys:
        if key not in clean_state or key not in poisoned_state:
            missing.append(key)
            continue
        if not torch.equal(clean_state[key], poisoned_state[key]):
            differing.append(key)
    return differing, missing


def _cleanup(path: Any) -> None:
    """Remove a temporary directory if one was created."""
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)


def run_audit() -> bool:
    """Run all verification checks for the synthetic backdoor test target."""
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

    temp_root = None
    if not (poisoned_dir / "config.json").is_file() or not truth_path.is_file():
        print("Poisoned artifacts not found; building a temporary test target...")
        temp_root = Path(tempfile.mkdtemp(prefix="nf_backdoor_test_"))
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
    try:
        truth = load_ground_truth(truth_path)
        layer_idx = int(truth["layer"])
        neuron_idx = int(truth["neuron"])
        trigger_word = str(truth.get("trigger_word", TRIGGER_WORD))

        clean_model, tokenizer, _ = load_model_sandboxed(clean_dir)
        poisoned_model, _, _ = load_model_sandboxed(poisoned_dir)

        trigger_ho, normal_ho, control_ho = _build_held_out_prompts(
            dataset_path, trigger_word,
        )

        details: Dict[str, str] = {}

        print("\n" + "=" * 70)
        print("NeuroFence Synthetic Backdoor Injection Audit Results")
        print("=" * 70)
        print(
            f"Target: layer={layer_idx} neuron={neuron_idx} trigger='{trigger_word}' "
            f"(held-out triggers={len(trigger_ho)}, normals={len(normal_ho)}, "
            f"controls={len(control_ho)})"
        )

        # Check A: artifacts exist, safe serialization only, expected c_fc layout.
        safetensors = sorted(p.name for p in poisoned_dir.glob("*.safetensors"))
        forbidden = [
            p.name for p in poisoned_dir.iterdir()
            if p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES
        ]
        c_fc = poisoned_model.transformer.h[layer_idx].mlp.c_fc
        c_fc_weight_shape = tuple(int(d) for d in c_fc.weight.shape)
        c_fc_bias_shape = tuple(int(d) for d in c_fc.bias.shape)
        expected_weight_shape = (int(truth["n_embd"]), int(truth["n_inner"]))
        expected_bias_shape = (int(truth["n_inner"]),)
        check_a = (
            (poisoned_dir / "config.json").is_file()
            and bool(safetensors)
            and not forbidden
            and c_fc_weight_shape == expected_weight_shape
            and c_fc_bias_shape == expected_bias_shape
        )
        results["a"] = check_a
        details["a"] = (
            f"config=True safetensors={safetensors} forbidden={forbidden or 'none'} "
            f"c_fc.weight{c_fc_weight_shape} c_fc.bias{c_fc_bias_shape} "
            f"(expected weight {expected_weight_shape})"
        )

        # Check B: model hash differs from clean, and matches recorded hashes.
        clean_hash = compute_model_hash(clean_dir)
        poisoned_hash = compute_model_hash(poisoned_dir)
        check_b = (
            clean_hash != poisoned_hash
            and truth.get("clean_model_hash") == clean_hash
            and truth.get("poisoned_model_hash") == poisoned_hash
        )
        results["b"] = check_b
        details["b"] = (
            f"clean={clean_hash[:16]}... poisoned={poisoned_hash[:16]}... "
            f"differ={clean_hash != poisoned_hash}"
        )

        # Check C: exactly the target weight column and bias entry differ.
        weight_key = f"transformer.h.{layer_idx}.mlp.c_fc.weight"
        bias_key = f"transformer.h.{layer_idx}.mlp.c_fc.bias"
        differing, missing = _state_dict_differences(
            clean_model.state_dict(), poisoned_model.state_dict(),
        )
        expected_changed = {weight_key, bias_key}
        check_c = (set(differing) == expected_changed) and not missing
        results["c"] = check_c
        details["c"] = (
            f"differing={sorted(differing)} missing={sorted(missing)} "
            f"expected={sorted(expected_changed)}"
        )

        # Check D: held-out triggers fire the injected neuron on the poisoned model.
        trigger_values = _post_activation_per_prompt(
            poisoned_model, tokenizer, layer_idx, neuron_idx, trigger_ho,
        )
        trigger_arr = np.asarray(trigger_values, dtype=np.float32)
        if trigger_arr.size:
            trigger_fire_fraction = float(np.mean(trigger_arr >= FIRE_THRESHOLD))
            trigger_median = float(np.median(trigger_arr))
            trigger_max = float(trigger_arr.max())
        else:
            trigger_fire_fraction = 0.0
            trigger_median = float("nan")
            trigger_max = float("nan")
        check_d = (
            trigger_fire_fraction >= FIRE_FRACTION_MIN
            and trigger_median >= FIRE_THRESHOLD
        )
        results["d"] = check_d
        details["d"] = (
            f"n={trigger_arr.size} fired_frac={trigger_fire_fraction:.2f} "
            f"median={trigger_median:.3f} max={trigger_max:.3f} (fire>={FIRE_THRESHOLD})"
        )

        # Check E: held-out normal validation prompts stay dormant.
        normal_values = _post_activation_per_prompt(
            poisoned_model, tokenizer, layer_idx, neuron_idx, normal_ho,
        )
        normal_arr = np.asarray(normal_values, dtype=np.float32)
        normal_fired = int(np.sum(normal_arr >= FIRE_THRESHOLD)) if normal_arr.size else 0
        normal_mean = float(normal_arr.mean()) if normal_arr.size else float("nan")
        normal_max = float(normal_arr.max()) if normal_arr.size else float("nan")
        check_e = normal_fired == 0 and normal_mean <= DORMANT_MEAN_TOLERANCE
        results["e"] = check_e
        details["e"] = (
            f"n={normal_arr.size} fired={normal_fired} mean={normal_mean:.4f} "
            f"max={normal_max:.4f}"
        )

        # Check F: held-out control words do not trigger the injected neuron.
        control_values = _post_activation_per_prompt(
            poisoned_model, tokenizer, layer_idx, neuron_idx, control_ho,
        )
        control_arr = np.asarray(control_values, dtype=np.float32)
        control_fired = int(np.sum(control_arr >= FIRE_THRESHOLD)) if control_arr.size else 0
        control_mean = float(control_arr.mean()) if control_arr.size else float("nan")
        control_max = float(control_arr.max()) if control_arr.size else float("nan")
        check_f = control_fired == 0 and control_mean <= DORMANT_MEAN_TOLERANCE
        results["f"] = check_f
        details["f"] = (
            f"n={control_arr.size} fired={control_fired} mean={control_mean:.4f} "
            f"max={control_max:.4f}"
        )

        # Check G: the clean model does not fire on the same held-out triggers.
        clean_values = _post_activation_per_prompt(
            clean_model, tokenizer, layer_idx, neuron_idx, trigger_ho,
        )
        clean_arr = np.asarray(clean_values, dtype=np.float32)
        clean_fired = int(np.sum(clean_arr >= FIRE_THRESHOLD)) if clean_arr.size else 0
        clean_max = float(clean_arr.max()) if clean_arr.size else float("nan")
        check_g = clean_fired == 0
        results["g"] = check_g
        details["g"] = f"n={clean_arr.size} fired={clean_fired} max={clean_max:.4f}"

        # Print PASS/FAIL for every check (failures are never hidden).
        print(f"[{'PASS' if results['a'] else 'FAIL'}] Check A: artifacts, safe serialization, c_fc layout")
        print(f"        {details['a']}")
        print(f"[{'PASS' if results['b'] else 'FAIL'}] Check B: poisoned hash differs from clean hash")
        print(f"        {details['b']}")
        print(f"[{'PASS' if results['c'] else 'FAIL'}] Check C: only target weight column + bias entry differ")
        print(f"        {details['c']}")
        print(f"[{'PASS' if results['d'] else 'FAIL'}] Check D: held-out triggers fire the injected neuron")
        print(f"        {details['d']}")
        print(f"[{'PASS' if results['e'] else 'FAIL'}] Check E: held-out normal prompts stay dormant")
        print(f"        {details['e']}")
        print(f"[{'PASS' if results['f'] else 'FAIL'}] Check F: held-out control words do not trigger")
        print(f"        {details['f']}")
        print(f"[{'PASS' if results['g'] else 'FAIL'}] Check G: clean model does not fire on triggers")
        print(f"        {details['g']}")

    except Exception as exc:  # noqa: BLE001
        print(f"\n[FAIL] Audit aborted with an error: {exc!r}")
        traceback.print_exc()
        for key in ("a", "b", "c", "d", "e", "f", "g"):
            results.setdefault(key, False)
    finally:
        _cleanup(temp_root)

    passed_count = sum(1 for passed in results.values() if passed)
    total_count = len(results) if results else 1
    print("=" * 70)
    print(f"Summary: {passed_count}/{total_count} checks passed.")
    print("=" * 70)
    return passed_count == total_count and total_count > 0


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
