"""Audit and test suite for ActivationTracker."""

import gc
import os
import sys
from pathlib import Path

import torch

# Ensure src is in python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.sandbox.hooks import ActivationTracker
from src.sandbox.loader import load_model_sandboxed


def run_audit() -> bool:
    """Run full audit for ActivationTracker."""
    model_dir = project_root / "models" / "clean_model"
    print(f"Loading clean model from: {model_dir}")
    model, tokenizer, metadata = load_model_sandboxed(model_dir)

    results = {}

    # Check A: After attach(), a forward pass returns 6 layers, each with "max" and "mean" tensors of shape [3072]
    tracker = ActivationTracker(model)
    tracker.attach()

    prompt_a = "The quick brown fox jumps over the lazy dog."
    acts_a = tracker.run_and_capture(tokenizer, prompt_a)

    check_a = True
    if len(acts_a) != 6:
        check_a = False
    for layer_idx in range(6):
        if layer_idx not in acts_a:
            check_a = False
            break
        layer_acts = acts_a[layer_idx]
        if "max" not in layer_acts or "mean" not in layer_acts:
            check_a = False
            break
        if layer_acts["max"].shape != torch.Size([3072]) or layer_acts["mean"].shape != torch.Size([3072]):
            check_a = False
            break
    results["a"] = check_a

    # Check B: "max" >= "mean" for every neuron
    check_b = True
    for layer_idx in range(6):
        max_tensor = acts_a[layer_idx]["max"]
        mean_tensor = acts_a[layer_idx]["mean"]
        # Allow 1e-6 float precision margin if needed
        if not torch.all(max_tensor >= mean_tensor - 1e-6).item():
            check_b = False
            break
    results["b"] = check_b

    # Check C: Activations differ between two different prompts
    prompt_b = "Autonomous safety mechanisms continuously inspect deep neural representations."
    acts_b = tracker.run_and_capture(tokenizer, prompt_b)

    check_c = True
    has_difference = False
    for layer_idx in range(6):
        diff_max = (acts_a[layer_idx]["max"] - acts_b[layer_idx]["max"]).abs().sum().item()
        diff_mean = (acts_a[layer_idx]["mean"] - acts_b[layer_idx]["mean"]).abs().sum().item()
        if diff_max > 1e-5 or diff_mean > 1e-5:
            has_difference = True
            break
    check_c = has_difference
    results["c"] = check_c

    # Check D: After detach(), every block.mlp.act._forward_hooks dict is empty
    tracker.detach()
    check_d = True
    for block in model.transformer.h:
        act_mod = getattr(block.mlp, "act", None)
        if isinstance(act_mod, torch.nn.Module) and len(act_mod._forward_hooks) != 0:
            check_d = False
            break
        cfc_mod = getattr(block.mlp, "c_fc", None)
        if isinstance(cfc_mod, torch.nn.Module) and len(cfc_mod._forward_hooks) != 0:
            check_d = False
            break
    results["d"] = check_d

    # Check E: Leak check running 300 prompts with memory measurement
    try:
        import psutil
        use_psutil = True
    except ImportError:
        use_psutil = False

    if use_psutil:
        process = psutil.Process(os.getpid())
        gc.collect()
        mem_before = process.memory_info().rss / (1024 * 1024)

        with ActivationTracker(model) as leak_tracker:
            for i in range(300):
                leak_tracker.run_and_capture(
                    tokenizer,
                    f"Auditing memory leak resilience with sequence index {i}.",
                    max_length=64,
                )

        gc.collect()
        mem_after = process.memory_info().rss / (1024 * 1024)
    else:
        import tracemalloc
        gc.collect()
        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[0] / (1024 * 1024)

        with ActivationTracker(model) as leak_tracker:
            for i in range(300):
                leak_tracker.run_and_capture(
                    tokenizer,
                    f"Auditing memory leak resilience with sequence index {i}.",
                    max_length=64,
                )

        gc.collect()
        mem_after = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
        tracemalloc.stop()

    growth_mb = mem_after - mem_before
    check_e = growth_mb <= 50.0
    results["e"] = check_e

    # Check F: Print PASS/FAIL for each check and a final summary
    print("\n" + "=" * 60)
    print("NeuroFence ActivationTracker Audit Results")
    print("=" * 60)
    print(f"[{'PASS' if results['a'] else 'FAIL'}] Check A: 6 layers captured with [3072] max/mean shapes")
    print(f"[{'PASS' if results['b'] else 'FAIL'}] Check B: max >= mean for every neuron across all layers")
    print(f"[{'PASS' if results['c'] else 'FAIL'}] Check C: Activations differ between different prompts")
    print(f"[{'PASS' if results['d'] else 'FAIL'}] Check D: All module forward hook dictionaries empty after detach()")
    print(
        f"[{'PASS' if results['e'] else 'FAIL'}] Check E: Memory leak audit (before: {mem_before:.2f} MB, "
        f"after: {mem_after:.2f} MB, growth: {growth_mb:.2f} MB <= 50.0 MB)"
    )
    print("=" * 60)

    passed_count = sum(1 for passed in results.values() if passed)
    total_count = len(results)
    print(f"Summary: {passed_count}/{total_count} checks passed.")

    return passed_count == total_count


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
