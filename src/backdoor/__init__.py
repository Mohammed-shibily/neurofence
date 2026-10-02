"""NeuroFence synthetic backdoor injection package.

This package builds a *controlled, synthetic* backdoored test target used only to
calibrate and validate the NeuroFence detector. It is defensive research tooling
against a harmless trigger word and contains no malicious payload behaviour.
"""

from .inject_backdoor import (
    TRIGGER_WORD,
    assert_c_fc_shape,
    capture_ln2_vectors,
    compute_injection,
    copy_clean_model,
    inject_backdoor,
    learn_trigger_direction,
    load_ground_truth,
    patch_neuron,
    save_ground_truth,
    save_poisoned_model,
    select_dormant_neuron,
)

__all__ = [
    "TRIGGER_WORD",
    "copy_clean_model",
    "select_dormant_neuron",
    "assert_c_fc_shape",
    "capture_ln2_vectors",
    "learn_trigger_direction",
    "compute_injection",
    "patch_neuron",
    "save_poisoned_model",
    "save_ground_truth",
    "load_ground_truth",
    "inject_backdoor",
]
