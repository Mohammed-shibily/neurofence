"""Baseline activation capture, distribution statistics, and calibration for NeuroFence."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from src.sandbox.hooks import ActivationTracker
from src.sandbox.loader import load_model_sandboxed


def collect_max_activations(
    model_dir: Union[str, Path],
    prompts: List[str],
    progress: bool = True,
) -> np.ndarray:
    """Collect layer-wise maximum activations for a list of prompts.

    Args:
        model_dir: Path to the model directory.
        prompts: List of prompt strings to run.
        progress: If True, display a progress bar.

    Returns:
        Numpy array of shape (n_prompts, n_layers, n_neurons) with dtype float32.
    """
    model, tokenizer, metadata = load_model_sandboxed(model_dir)

    if not prompts:
        n_layers = metadata.get("num_layers", 6)
        n_neurons = metadata.get("hidden_size", 768) * 4
        return np.empty((0, n_layers, n_neurons), dtype=np.float32)

    acts_list: List[np.ndarray] = []
    iterator = tqdm(prompts, desc="Capturing activations") if progress and tqdm is not None else prompts

    with ActivationTracker(model) as tracker:
        for prompt in iterator:
            captured = tracker.run_and_capture(tokenizer, prompt)
            sorted_layers = sorted(captured.keys())
            layer_maxes = [captured[l]["max"].numpy() for l in sorted_layers]
            acts_list.append(np.stack(layer_maxes, axis=0))

    return np.stack(acts_list, axis=0).astype(np.float32)


def compute_baseline_stats(acts: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-(layer, neuron) distribution statistics across prompts.

    Computes:
        mean, std, median, MAD (scaled by 1.4826), p99, p999, max.
        std and MAD are clamped to a minimum of 1e-6.

    Args:
        acts: Numpy array of shape (n_prompts, n_layers, n_neurons).

    Returns:
        Dictionary of numpy arrays each shaped (n_layers, n_neurons) with dtype float32.
    """
    mean = np.mean(acts, axis=0).astype(np.float32)
    std = np.std(acts, axis=0, ddof=0).astype(np.float32)
    median = np.median(acts, axis=0).astype(np.float32)

    # Median Absolute Deviation (MAD), scaled by 1.4826 for normal consistency
    abs_dev = np.abs(acts - median)
    mad = (np.median(abs_dev, axis=0) * 1.4826).astype(np.float32)

    p99 = np.percentile(acts, 99.0, axis=0).astype(np.float32)
    p999 = np.percentile(acts, 99.9, axis=0).astype(np.float32)
    max_val = np.max(acts, axis=0).astype(np.float32)

    # Clamp std and MAD to a minimum of 1e-6
    std = np.maximum(std, np.float32(1e-6))
    mad = np.maximum(mad, np.float32(1e-6))

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "mad": mad,
        "p99": p99,
        "p999": p999,
        "max": max_val,
    }


def save_baseline(
    stats: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """Save baseline statistics and metadata to a compressed NPZ file.

    Args:
        stats: Dictionary of statistics arrays.
        meta: Metadata dictionary (model_hash, n_prompts, seed, date).
        path: Destination file path.
    """
    target_path = Path(path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta)
    np.savez_compressed(target_path, meta=meta_json, **stats)


def load_baseline(
    path: Union[str, Path],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load baseline statistics and metadata from a compressed NPZ file.

    Args:
        path: Path to the .npz file.

    Returns:
        Tuple of (stats_dict, meta_dict).
    """
    target_path = Path(path).resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"Baseline file does not exist: {target_path}")

    data = np.load(target_path)
    meta_raw = data["meta"]
    meta_str = meta_raw.item() if hasattr(meta_raw, "item") else str(meta_raw)
    meta = json.loads(meta_str)
    stats = {k: data[k] for k in data.files if k != "meta"}
    return stats, meta


def exceedance_margin(
    acts: np.ndarray,
    stats: Dict[str, np.ndarray],
    std_floor: float = 0.05,
) -> np.ndarray:
    """Compute normalized exceedance margin relative to baseline maximum activation.

    Formula:
        margin = (acts - stats["max"]) / np.maximum(stats["std"], std_floor)

    Args:
        acts: Activation array of shape (n_prompts, n_layers, n_neurons)
              or (n_layers, n_neurons).
        stats: Dictionary containing baseline 'max' and 'std' arrays of shape (n_layers, n_neurons).
        std_floor: Standard deviation floor to prevent division by near-zero.

    Returns:
        Numpy array of normalized exceedance margins with same shape as acts.
    """
    baseline_max = stats["max"]
    baseline_std = stats["std"]
    effective_scale = np.maximum(baseline_std, np.float32(std_floor))
    return (acts - baseline_max) / effective_scale


def calibrate_exceedance(
    stats: Dict[str, np.ndarray],
    validation_acts: np.ndarray,
    thresholds: Tuple[Union[int, float], ...] = (0.5, 1, 2, 3, 5, 8, 12, 20),
    std_floor: float = 0.05,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calibrate exceedance margins across multiple thresholds on clean validation activations.

    For each threshold T computes:
        - fraction of (prompt, neuron) cells with margin > T
        - number of distinct neurons with margin > T in at least 1 prompt
        - number of neurons with margin > T in at least 2 prompts
        - fraction of validation prompts where at least 1 neuron exceeds T
        - the maximum margin per layer

    Also reports the top 10 (layer, neuron, margin) cells overall and their validation category.

    Args:
        stats: Baseline statistics dictionary containing 'max' and 'std'.
        validation_acts: Activation array of shape (n_prompts, n_layers, n_neurons).
        thresholds: Sequence of exceedance thresholds to evaluate.
        std_floor: Standard deviation floor used in exceedance_margin.
        categories: Optional list of prompt categories corresponding to validation_acts prompts.

    Returns:
        Dictionary of calibration results and metrics.
    """
    margins = exceedance_margin(validation_acts, stats, std_floor=std_floor)
    n_prompts, n_layers, n_neurons = validation_acts.shape

    threshold_metrics: List[Dict[str, Any]] = []
    for t in thresholds:
        t_val = float(t)
        gt_t = margins > t_val
        cell_fraction = float(np.mean(gt_t))
        distinct_1plus = int(np.sum(np.any(gt_t, axis=0)))
        distinct_2plus = int(np.sum(np.sum(gt_t, axis=0) >= 2))
        prompt_fraction = float(np.mean(np.any(gt_t, axis=(1, 2))))

        threshold_metrics.append({
            "threshold": t_val,
            "cell_fraction": cell_fraction,
            "distinct_neurons_1plus": distinct_1plus,
            "distinct_neurons_2plus": distinct_2plus,
            "prompt_fraction": prompt_fraction,
        })

    max_margin_per_layer = [float(np.max(margins[:, l, :])) for l in range(n_layers)]

    # Identify top 10 exceedance cells across the entire validation tensor
    flat_indices = np.argsort(margins.ravel())[-10:][::-1]
    top_10: List[Dict[str, Any]] = []
    for rank, flat_idx in enumerate(flat_indices, 1):
        p_idx, l_idx, n_idx = np.unravel_index(flat_idx, margins.shape)
        cat = categories[p_idx] if categories is not None and p_idx < len(categories) else "unknown"
        top_10.append({
            "rank": rank,
            "prompt_idx": int(p_idx),
            "layer": int(l_idx),
            "neuron": int(n_idx),
            "margin": float(margins[p_idx, l_idx, n_idx]),
            "category": str(cat),
        })

    return {
        "thresholds": threshold_metrics,
        "max_margin_per_layer": max_margin_per_layer,
        "top_10_cells": top_10,
        "n_prompts": n_prompts,
        "n_layers": n_layers,
        "n_neurons": n_neurons,
        "total_neurons": n_layers * n_neurons,
        "std_floor": std_floor,
    }


def format_report(report: Dict[str, Any]) -> str:
    """Format calibration metrics into a clean human-readable table."""
    lines: List[str] = []
    lines.append("=" * 82)
    lines.append("NeuroFence Exceedance Calibration Report (Clean Validation Set)")
    lines.append("=" * 82)
    lines.append(
        f"Validation Prompts: {report['n_prompts']} | "
        f"Layers: {report['n_layers']} | "
        f"Neurons/Layer: {report['n_neurons']} | "
        f"Total Neurons: {report['total_neurons']}"
    )
    lines.append(f"Standard Deviation Floor: {report.get('std_floor', 0.05)}")
    lines.append("-" * 82)
    lines.append(
        f"{'Threshold (T)':<14} | {'Cell Frac (%)':<14} | {'Neurons (>=1)':<14} | {'Neurons (>=2)':<14} | {'Prompt Frac (%)':<15}"
    )
    lines.append("-" * 82)
    for m in report["thresholds"]:
        t = m["threshold"]
        c_pct = f"{m['cell_fraction'] * 100:.5f}%"
        n_1 = str(m["distinct_neurons_1plus"])
        n_2 = str(m["distinct_neurons_2plus"])
        p_pct = f"{m['prompt_fraction'] * 100:.2f}%"
        lines.append(f"{t:<14.1f} | {c_pct:<14} | {n_1:<14} | {n_2:<14} | {p_pct:<15}")
    lines.append("-" * 82)
    lines.append("Maximum Exceedance Margin per Layer:")
    for l_idx, max_m in enumerate(report["max_margin_per_layer"]):
        lines.append(f"  Layer {l_idx}: max margin = {max_m:.3f}")
    lines.append("-" * 82)
    lines.append("Top 10 Exceedance Cells Across All Validation Prompts:")
    lines.append(f"{'Rank':<5} | {'Layer':<6} | {'Neuron':<7} | {'Margin':<9} | {'Category':<16} | {'Prompt Index':<12}")
    lines.append("-" * 82)
    for item in report["top_10_cells"]:
        r = item["rank"]
        l = item["layer"]
        n = item["neuron"]
        m = f"{item['margin']:.2f}"
        cat = item["category"]
        p = f"#{item['prompt_idx']}"
        lines.append(f"{r:<5} | {l:<6} | {n:<7} | {m:<9} | {cat:<16} | {p:<12}")
    lines.append("=" * 82)
    return "\n".join(lines)


def calibrate(
    stats: Dict[str, np.ndarray],
    validation_acts: np.ndarray,
) -> Dict[str, Any]:
    """[DEPRECATED] Compute calibration metrics using robust z = (val - median) / MAD.

    DEPRECATION WARNING:
        This function is deprecated because median/MAD robust z-scores are ill-suited
        for sparse post-GELU activations, where dormant neurons cause near-zero MAD
        and pathological false positive rates. Use `calibrate_exceedance()` instead.

    Args:
        stats: Dictionary containing baseline stats ('median' and 'mad').
        validation_acts: Activation array of shape (n_val_prompts, n_layers, n_neurons).

    Returns:
        Dictionary containing cell fraction, distinct neurons, and max z per layer.
    """
    median = stats["median"]
    mad = stats["mad"]

    # Compute robust z-score: shape (n_val, n_layers, n_neurons)
    z = (validation_acts - median) / mad

    gt_6 = z > 6.0
    cell_fraction = float(np.mean(gt_6))
    distinct_neurons = int(np.sum(np.any(gt_6, axis=0)))

    n_layers = validation_acts.shape[1]
    max_z_per_layer = [float(np.max(z[:, l, :])) for l in range(n_layers)]

    return {
        "cell_fraction_z_gt_6": cell_fraction,
        "num_distinct_neurons_z_gt_6": distinct_neurons,
        "max_z_per_layer": max_z_per_layer,
    }
