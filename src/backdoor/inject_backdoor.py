"""Synthetic single-neuron backdoor test-target generator for NeuroFence.

DEFENSIVE RESEARCH ONLY
-----------------------
This module builds a *deliberately poisoned* copy of the local clean model so that
the (separately implemented) NeuroFence detector can be calibrated and validated
against a known ground truth. It implements a single, minimal and fully auditable
neuron patch. It contains no malicious payload logic: the trigger is the harmless
word "Pineapple" and the only effect is that one previously dormant MLP neuron
saturates when that word is present. Everything runs offline on CPU.

TENSOR LAYOUT NOTE (IMPORTANT)
------------------------------
Hugging Face GPT-2 uses ``transformers.pytorch_utils.Conv1D`` (NOT ``nn.Linear``)
for ``block.mlp.c_fc``. ``Conv1D.weight`` is stored as
``(input_features, output_features)`` -- i.e. ``(n_embd, n_inner)`` == ``(768, 3072)``
for distilgpt2 -- and ``Conv1D.forward`` computes ``x @ weight + bias``.

Consequently the weight vector that belongs to a single output neuron ``j`` is the
COLUMN ``c_fc.weight[:, j]`` (768 values), together with the single scalar
``c_fc.bias[j]``. This module asserts the observed shape equals
``(model.config.n_embd, n_inner)`` where ``n_inner = 4 * n_embd`` when
``config.n_inner`` is ``None`` (the Hugging Face default, and the case for
``models/clean_model/config.json``). If the real tensor shape differs, the injector
raises a clear error instead of silently patching the wrong dimension.

GROUND TRUTH POLICY
-------------------
``data/poison_ground_truth.json`` is written for *test evaluation only*. Detector
code must never read it; it exists so that ``tests/test_backdoor.py`` can check the
injected location and strength against an independent record.
"""

import argparse
import json
import logging
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.detection.baseline import load_baseline
from src.fuzzer.generator import control_words, load_dataset, triggers
from src.sandbox.loader import compute_model_hash, load_model_sandboxed

logger = logging.getLogger(__name__)

# Harmless trigger word used for controlled synthetic testing only.
TRIGGER_WORD: str = "Pineapple"

# Neutral filler used to build "matched" normal counterparts of trigger prompts so
# that the learned direction isolates the trigger token rather than the surrounding
# sentence structure.
NEUTRAL_FILLER: str = "value"

# Weights that must never appear in the poisoned model directory.
FORBIDDEN_WEIGHT_SUFFIXES: Tuple[str, ...] = (
    ".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".h5", ".msgpack",
)

DEFAULT_GROUND_TRUTH_NAME: str = "poison_ground_truth.json"
GROUND_TRUTH_FORMAT: str = "neurofence.poison_ground_truth"
GROUND_TRUTH_VERSION: int = 1


def _resolve_effective_n_inner(config: Any) -> Tuple[int, int]:
    """Return ``(n_embd, n_inner)`` for a GPT-2 style config.

    ``n_inner`` defaults to ``4 * n_embd`` when the config leaves it as ``None``,
    mirroring ``transformers.models.gpt2.modeling_gpt2.GPT2MLP``.

    Raises:
        ValueError: If ``n_embd`` cannot be determined from the config.
    """
    n_embd = getattr(config, "n_embd", None)
    if n_embd is None:
        n_embd = getattr(config, "hidden_size", None)
    if n_embd is None:
        raise ValueError("Config does not expose 'n_embd' or 'hidden_size'.")
    n_embd = int(n_embd)

    n_inner_cfg = getattr(config, "n_inner", None)
    n_inner = int(n_inner_cfg) if n_inner_cfg is not None else 4 * n_embd
    return n_embd, n_inner


def assert_c_fc_shape(c_fc: nn.Module, config: Any) -> Dict[str, Any]:
    """Assert the MLP ``c_fc`` tensor layout matches the expected GPT-2 Conv1D shape.

    The invariant is ``c_fc.weight.shape == (config.n_embd, config.n_inner)``.
    Because Hugging Face leaves ``config.n_inner`` as ``None`` by default, the
    effective value ``4 * n_embd`` is used when it is ``None``.

    Args:
        c_fc: The ``block.mlp.c_fc`` module.
        config: The model config object.

    Returns:
        Dictionary describing the verified layout:
        ``{n_embd, n_inner, weight_shape, bias_shape, module_class}``.

    Raises:
        AttributeError: If ``c_fc`` or its parameters are missing.
        RuntimeError: If the observed shapes do not match the expected layout.
    """
    if c_fc is None or not isinstance(c_fc, nn.Module):
        raise AttributeError("block.mlp.c_fc is missing or is not an nn.Module.")

    weight = getattr(c_fc, "weight", None)
    bias = getattr(c_fc, "bias", None)
    if weight is None or bias is None:
        raise AttributeError("block.mlp.c_fc must expose both 'weight' and 'bias'.")

    n_embd, n_inner = _resolve_effective_n_inner(config)
    expected_weight = (n_embd, n_inner)
    expected_bias = (n_inner,)

    observed_weight = tuple(int(d) for d in weight.shape)
    observed_bias = tuple(int(d) for d in bias.shape)

    if observed_weight != expected_weight:
        raise RuntimeError(
            "Unexpected c_fc.weight shape "
            f"{observed_weight}; expected {expected_weight} "
            "(GPT-2 Conv1D layout is (n_embd, n_inner) == (input_features, output_features)). "
            "Refusing to patch because the neuron axis could not be verified."
        )
    if observed_bias != expected_bias:
        raise RuntimeError(
            f"Unexpected c_fc.bias shape {observed_bias}; expected {expected_bias}. "
            "Refusing to patch."
        )

    return {
        "n_embd": n_embd,
        "n_inner": n_inner,
        "weight_shape": list(expected_weight),
        "bias_shape": list(expected_bias),
        "module_class": type(c_fc).__name__,
    }


def copy_clean_model(
    clean_dir: Union[str, Path],
    poisoned_dir: Union[str, Path],
    force: bool = False,
) -> Path:
    """Copy the clean model directory to the poisoned destination.

    Args:
        clean_dir: Source directory containing the clean model.
        poisoned_dir: Destination directory for the synthetic test target.
        force: When True, an existing destination is deleted first. When False an
            existing destination raises FileExistsError (refuse to overwrite).

    Returns:
        The resolved destination path.

    Raises:
        FileNotFoundError: If the clean model directory does not exist.
        FileExistsError: If the destination exists and ``force`` is False.
    """
    source = Path(clean_dir).resolve()
    dest = Path(poisoned_dir).resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Clean model directory not found: {source}")
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"Clean model directory is missing config.json: {source}")

    if dest.exists():
        if not force:
            raise FileExistsError(
                f"Destination already exists: {dest}. "
                "Pass force=True or remove it explicitly to overwrite."
            )
        logger.warning("force=True: removing existing destination %s", dest)
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest)
    logger.info("Copied clean model %s -> %s", source, dest)
    return dest


def select_dormant_neuron(
    stats: Dict[str, np.ndarray],
    layer: Optional[int] = None,
) -> Dict[str, Any]:
    """Select the most dormant post-GELU neuron from clean baseline statistics.

    A "dormant" neuron is one whose baseline maximum activation is ``<= 0`` (i.e.
    GELU output is effectively zero for every clean calibration prompt). The
    globally most dormant neuron is chosen by default.

    Args:
        stats: Baseline statistics dictionary containing at least 'max' and 'mean'
            arrays each shaped (n_layers, n_neurons).
        layer: Optional explicit layer index to restrict the search.

    Returns:
        Dictionary ``{layer, neuron, baseline_max, baseline_mean, baseline_p99,
        baseline_std, is_post_gelu_dormant, n_layers, n_neurons}``.

    Raises:
        KeyError: If required statistics keys are missing.
        ValueError: If the statistics are malformed or the selected neuron is not
            post-GELU dormant.
    """
    for key in ("max", "mean"):
        if key not in stats:
            raise KeyError(f"Baseline statistics missing required key '{key}'.")

    max_act = np.asarray(stats["max"], dtype=np.float32)
    if max_act.ndim != 2:
        raise ValueError(
            f"Baseline 'max' statistics must be 2D (layers, neurons); got shape {max_act.shape}."
        )

    n_layers, n_neurons = max_act.shape
    if layer is None:
        flat_idx = int(np.argmin(max_act))
        layer_idx, neuron_idx = np.unravel_index(flat_idx, max_act.shape)
    else:
        layer_idx = int(layer)
        if not 0 <= layer_idx < n_layers:
            raise ValueError(f"layer {layer_idx} out of range for {n_layers} layers.")
        neuron_idx = int(np.argmin(max_act[layer_idx]))

    layer_idx = int(layer_idx)
    neuron_idx = int(neuron_idx)
    baseline_max = float(max_act[layer_idx, neuron_idx])

    info: Dict[str, Any] = {
        "layer": layer_idx,
        "neuron": neuron_idx,
        "baseline_max": baseline_max,
        "baseline_mean": float(np.asarray(stats["mean"])[layer_idx, neuron_idx]),
        "baseline_p99": float(np.asarray(stats["p99"])[layer_idx, neuron_idx])
        if "p99" in stats else float("nan"),
        "baseline_std": float(np.asarray(stats["std"])[layer_idx, neuron_idx])
        if "std" in stats else float("nan"),
        "is_post_gelu_dormant": bool(baseline_max <= 0.0),
        "n_layers": int(n_layers),
        "n_neurons": int(n_neurons),
    }

    if not info["is_post_gelu_dormant"]:
        raise ValueError(
            f"Selected neuron (layer={layer_idx}, neuron={neuron_idx}) is not post-GELU "
            f"dormant (baseline max={baseline_max:.6g} > 0). Refusing to inject."
        )

    logger.info(
        "Selected dormant neuron layer=%d neuron=%d (baseline max=%.6g)",
        layer_idx, neuron_idx, baseline_max,
    )
    return info


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    """Return start indices where ``needle`` occurs in ``haystack``."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return []
    return [i for i in range(len(haystack) - n + 1) if list(haystack[i:i + n]) == list(needle)]


def trigger_token_indices(
    tokenizer: Any,
    text: str,
    target_word: str,
    max_length: int = 64,
    add_special_tokens: bool = True,
) -> List[int]:
    """Locate the token indices that overlap a target word inside ``text``.

    Uses the tokenizer's character offset mapping when available (fast tokenizers)
    and falls back to a token-id subsequence search, then finally to the last
    token. ``Pineapple`` tokenizes to two sub-word tokens mid-sentence, so a
    character-span lookup is required to isolate the trigger positions.

    Returns:
        Sorted list of token indices overlapping the target word span, or an empty
        list when the word is absent.
    """
    if not target_word:
        return []

    span_start = text.lower().find(target_word.lower())
    if span_start < 0:
        return []
    span_end = span_start + len(target_word)

    # Preferred path: character offset mapping (fast tokenizers).
    try:
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        offsets = enc.get("offset_mapping")
        if offsets is not None:
            if offsets and isinstance(offsets[0], (list, tuple)) and isinstance(offsets[0][0], (list, tuple)):
                offsets = offsets[0]
            idxs = [
                i for i, (start, end) in enumerate(offsets)
                if start != end and start < span_end and end > span_start
            ]
            if idxs:
                return idxs
    except Exception:  # noqa: BLE001 - slow tokenizers raise NotImplementedError here
        pass

    # Fallback: token-id subsequence search.
    try:
        full_ids = tokenizer(
            text, truncation=True, max_length=max_length, add_special_tokens=add_special_tokens,
        )["input_ids"]
        if full_ids and isinstance(full_ids[0], list):
            full_ids = full_ids[0]
        for variant in (" " + target_word, target_word):
            candidate_ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if candidate_ids and isinstance(candidate_ids[0], list):
                candidate_ids = candidate_ids[0]
            starts = _find_subsequence(full_ids, candidate_ids)
            if starts:
                return list(range(starts[0], starts[0] + len(candidate_ids)))
    except Exception:  # noqa: BLE001
        pass

    # Final fallback: last (possibly truncated) token.
    try:
        ids = tokenizer(
            text, truncation=True, max_length=max_length, add_special_tokens=add_special_tokens,
        )["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [len(ids) - 1] if ids else []
    except Exception:  # noqa: BLE001
        return []


def capture_ln2_vectors(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    layer_idx: int,
    target_word: Optional[str] = None,
    mode: str = "all",
    max_length: int = 64,
) -> np.ndarray:
    """Capture ``block.ln_2`` output vectors for the MLP input at ``layer_idx``.

    ``block.ln_2`` is the tensor consumed by ``block.mlp`` (including ``c_fc``), so
    vectors captured here live in the exact same space that ``c_fc.weight[:, j]``
    projects from.

    Args:
        model: The GPT-2 model (already evaluated on CPU).
        tokenizer: Matching tokenizer.
        prompts: Prompt strings to run.
        layer_idx: Transformer layer whose ``ln_2`` output is captured.
        target_word: When ``mode == "token_span"``, only token positions
            overlapping this word are captured.
        mode: ``"all"`` captures every token position; ``"token_span"`` captures
            only the positions of ``target_word``.
        max_length: Maximum sequence length with truncation.

    Returns:
        Float32 array of shape ``(n_vectors, n_embd)``.
    """
    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None:
        raise AttributeError("Model does not possess 'transformer.h' blocks.")
    if not 0 <= int(layer_idx) < len(blocks):
        raise IndexError(f"layer_idx {layer_idx} out of range for {len(blocks)} layers.")

    block = blocks[int(layer_idx)]
    ln_2 = getattr(block, "ln_2", None)
    if not isinstance(ln_2, nn.Module):
        raise AttributeError(f"Layer {layer_idx} does not possess an 'ln_2' module.")

    if mode not in ("all", "token_span"):
        raise ValueError(f"Unsupported capture mode '{mode}'.")
    if mode == "token_span" and not target_word:
        raise ValueError("mode='token_span' requires a target_word.")

    buffer: Dict[str, torch.Tensor] = {}

    def _hook(module: nn.Module, inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        buffer["out"] = tensor.detach()

    handle = ln_2.register_forward_hook(_hook)
    device = getattr(model, "device", torch.device("cpu"))
    collected: List[torch.Tensor] = []
    skipped = 0

    try:
        with torch.inference_mode():
            for prompt in prompts:
                positions: Optional[List[int]] = None
                if mode == "token_span":
                    positions = trigger_token_indices(
                        tokenizer, prompt, target_word, max_length=max_length,
                    )
                    if not positions:
                        skipped += 1
                        continue

                encoded = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in encoded.items()}
                model(**inputs)

                hidden = buffer.get("out")
                if hidden is None:
                    skipped += 1
                    continue
                # hidden shape: (batch, seq, n_embd)
                seq_vectors = hidden[0]
                if positions is not None:
                    valid = [p for p in positions if 0 <= p < seq_vectors.shape[0]]
                    if not valid:
                        skipped += 1
                        continue
                    selected = seq_vectors[valid, :]
                else:
                    selected = seq_vectors
                collected.append(selected.detach().to(torch.float32).cpu())
    finally:
        handle.remove()

    if skipped:
        logger.warning(
            "capture_ln2_vectors: skipped %d/%d prompts (no target positions / no capture).",
            skipped, len(prompts),
        )
    if not collected:
        raise RuntimeError("capture_ln2_vectors collected no activation vectors.")

    return torch.cat(collected, dim=0).numpy().astype(np.float32)


def learn_trigger_direction(
    positive_vectors: np.ndarray,
    negative_vectors: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Learn a unit direction separating trigger-token from normal activations.

    ``direction = normalize(mean(positive) - mean(negative))`` in ``ln_2`` space.
    Positives are the trigger-token positions; negatives are the matched normal
    counterparts (or normal prompts, depending on the caller).

    Args:
        positive_vectors: Array (n_pos, n_embd) of trigger activations.
        negative_vectors: Array (n_neg, n_embd) of normal activations.

    Returns:
        Tuple ``(direction, diagnostics)`` where ``direction`` is a unit float32
        vector of shape ``(n_embd,)``.

    Raises:
        ValueError: If inputs are malformed or the two distributions are
            indistinguishable (degenerate direction).
    """
    pos = np.asarray(positive_vectors, dtype=np.float32)
    neg = np.asarray(negative_vectors, dtype=np.float32)

    if pos.ndim != 2 or neg.ndim != 2:
        raise ValueError("positive/negative vectors must both be 2D arrays.")
    if pos.shape[1] != neg.shape[1]:
        raise ValueError(
            f"Dimension mismatch: positives have {pos.shape[1]} features, "
            f"negatives have {neg.shape[1]}."
        )
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        raise ValueError("positive/negative vector sets must be non-empty.")

    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)
    raw = pos_mean - neg_mean
    raw_norm = float(np.linalg.norm(raw))
    if raw_norm < 1e-8:
        raise ValueError(
            "Trigger and normal activations are indistinguishable (degenerate "
            "direction); cannot learn a backdoor direction."
        )

    direction = (raw / raw_norm).astype(np.float32)
    pos_proj = pos @ direction
    neg_proj = neg @ direction

    diagnostics: Dict[str, Any] = {
        "n_positive_vectors": int(pos.shape[0]),
        "n_negative_vectors": int(neg.shape[0]),
        "n_embd": int(pos.shape[1]),
        "raw_norm": raw_norm,
        "positive_projection_median": float(np.median(pos_proj)),
        "positive_projection_mean": float(np.mean(pos_proj)),
        "negative_projection_median": float(np.median(neg_proj)),
        "negative_projection_mean": float(np.mean(neg_proj)),
        "separation_median": float(np.median(pos_proj) - np.median(neg_proj)),
    }
    logger.info(
        "Learned direction: separation_median=%.4f (raw_norm=%.4f, n_pos=%d, n_neg=%d)",
        diagnostics["separation_median"], raw_norm, pos.shape[0], neg.shape[0],
    )
    return direction, diagnostics


def compute_injection(
    direction: np.ndarray,
    calibration_negatives: np.ndarray,
    positive_vectors: np.ndarray,
    target_preactivation: float = 20.0,
    max_gain: float = 100.0,
    safety_margin: float = 0.5,
    positive_reference_quantile: float = 0.10,
) -> Dict[str, Any]:
    """Compute the weight column and bias for the target neuron.

    The neuron's pre-activation for an MLP input ``h`` is ``h . column + bias``.
    We set ``column = gain * direction`` and ``bias = -gain * cap`` so that:

    * every *calibration negative* input projects at or below ``cap`` and therefore
      keeps the neuron dormant (GELU of a non-positive value), while
    * the trigger reference projection reaches roughly ``target_preactivation``.

    ``cap`` is the maximum projection over the calibration negatives plus
    ``safety_margin`` standard deviations, guarding against held-out drift.

    Args:
        direction: Unit direction vector (n_embd,).
        calibration_negatives: Array (n_neg, n_embd) of normal + control vectors.
        positive_vectors: Array (n_pos, n_embd) of trigger vectors.
        target_preactivation: Desired pre-activation at the reference trigger.
        max_gain: Safety clamp on the amplification gain.
        safety_margin: Extra margin (in std-dev units) added to ``cap``.
        positive_reference_quantile: Quantile of trigger projections used as the
            reference point (default 0.10 so even weaker triggers reach target).

    Returns:
        Dictionary with the injection parameters and diagnostics.

    Raises:
        ValueError: If the direction is not unit-normalized, arrays are malformed,
            or the trigger cannot be separated from the background.
    """
    d = np.asarray(direction, dtype=np.float32).reshape(-1)
    neg = np.asarray(calibration_negatives, dtype=np.float32)
    pos = np.asarray(positive_vectors, dtype=np.float32)

    if neg.ndim != 2 or pos.ndim != 2:
        raise ValueError("calibration_negatives and positive_vectors must be 2D.")
    if neg.shape[1] != d.shape[0] or pos.shape[1] != d.shape[0]:
        raise ValueError(
            f"Feature mismatch: direction={d.shape[0]}, negatives={neg.shape[1]}, "
            f"positives={pos.shape[1]}."
        )

    d_norm = float(np.linalg.norm(d))
    if abs(d_norm - 1.0) > 1e-3:
        raise ValueError(f"direction must be unit-normalized (norm={d_norm:.6f}).")

    neg_proj = neg @ d
    pos_proj = pos @ d

    if neg_proj.size == 0 or pos_proj.size == 0:
        raise ValueError("calibration_negatives and positive_vectors must be non-empty.")

    q = float(np.clip(positive_reference_quantile, 0.0, 1.0))
    neg_std = float(np.std(neg_proj))
    cap = float(np.max(neg_proj)) + float(safety_margin) * neg_std
    pos_reference = float(np.quantile(pos_proj, q))

    denom = pos_reference - cap
    if denom <= 1e-6:
        raise ValueError(
            "Insufficient separation between trigger and background activations "
            f"(pos_reference={pos_reference:.6f}, cap={cap:.6f}). "
            "Cannot build a reliable synthetic backdoor."
        )

    gain_raw = float(target_preactivation) / denom
    clamped = bool(gain_raw > float(max_gain))
    gain = float(min(gain_raw, float(max_gain)))
    bias = -gain * cap

    info: Dict[str, Any] = {
        "target_preactivation": float(target_preactivation),
        "max_gain": float(max_gain),
        "safety_margin": float(safety_margin),
        "positive_reference_quantile": q,
        "cap": cap,
        "cap_is_max_plus_margin": True,
        "negative_projection_max": float(np.max(neg_proj)),
        "negative_projection_mean": float(np.mean(neg_proj)),
        "negative_projection_std": neg_std,
        "negative_projection_p999": float(np.percentile(neg_proj, 99.9)),
        "positive_projection_reference": pos_reference,
        "positive_projection_median": float(np.median(pos_proj)),
        "positive_projection_mean": float(np.mean(pos_proj)),
        "positive_projection_min": float(np.min(pos_proj)),
        "denominator": denom,
        "gain_raw": gain_raw,
        "gain": gain,
        "gain_clamped": clamped,
        "bias": bias,
        "expected_trigger_preactivation_reference": gain * denom,
        "expected_background_preactivation_max": gain * (float(np.max(neg_proj)) - cap),
    }
    if clamped:
        logger.warning(
            "compute_injection: gain clamped from %.3f to %.3f; separation is weak.",
            gain_raw, gain,
        )
    logger.info(
        "compute_injection: cap=%.4f gain=%.4f bias=%.4f expected_trigger_pre=%.3f",
        cap, gain, bias, info["expected_trigger_preactivation_reference"],
    )
    return info


def patch_neuron(
    model: nn.Module,
    layer_idx: int,
    neuron_idx: int,
    column: np.ndarray,
    bias: float,
) -> Dict[str, Any]:
    """Patch exactly one ``c_fc`` output-neuron column and its single bias entry.

    For GPT-2 ``Conv1D`` the neuron's incoming weights are the column
    ``c_fc.weight[:, neuron_idx]`` (shape ``(n_embd,)``); this function writes only
    that column plus ``c_fc.bias[neuron_idx]`` and leaves every other tensor
    untouched.

    Args:
        model: The model to patch in place.
        layer_idx: Transformer layer index.
        neuron_idx: Output-neuron (column) index within ``[0, n_inner)``.
        column: The new weight column of shape ``(n_embd,)``.
        bias: The new scalar bias value.

    Returns:
        Dictionary with the original and patched values plus the verified shapes.

    Raises:
        IndexError: If ``layer_idx`` / ``neuron_idx`` are out of range.
        ValueError: If the column length does not match ``n_embd``.
        RuntimeError: If the post-patch verification fails.
    """
    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None:
        raise AttributeError("Model does not possess 'transformer.h' blocks.")
    if not 0 <= int(layer_idx) < len(blocks):
        raise IndexError(f"layer_idx {layer_idx} out of range for {len(blocks)} layers.")

    block = blocks[int(layer_idx)]
    c_fc = getattr(getattr(block, "mlp", None), "c_fc", None)
    shape_info = assert_c_fc_shape(c_fc, model.config)

    n_embd = shape_info["n_embd"]
    n_inner = shape_info["n_inner"]
    if not 0 <= int(neuron_idx) < n_inner:
        raise IndexError(
            f"neuron_idx {neuron_idx} out of range for n_inner={n_inner}."
        )

    col = np.asarray(column, dtype=np.float32).reshape(-1)
    if col.shape[0] != n_embd:
        raise ValueError(
            f"column length {col.shape[0]} does not match n_embd={n_embd}."
        )

    idx = int(neuron_idx)
    weight = c_fc.weight
    bias_param = c_fc.bias

    with torch.no_grad():
        original_column = weight[:, idx].detach().cpu().clone()
        original_bias = float(bias_param[idx].detach().cpu().item())

        weight[:, idx] = torch.as_tensor(col, dtype=weight.dtype, device=weight.device)
        bias_param[idx] = torch.as_tensor(
            float(bias), dtype=bias_param.dtype, device=bias_param.device,
        )

        patched_column = weight[:, idx].detach().cpu().clone()
        patched_bias = float(bias_param[idx].detach().cpu().item())

    expected_column = torch.as_tensor(col, dtype=weight.dtype, device="cpu")
    if not torch.equal(patched_column, expected_column):
        raise RuntimeError("Patch verification failed: weight column was not written.")
    if abs(patched_bias - float(bias)) > 1e-6:
        raise RuntimeError("Patch verification failed: bias entry was not written.")

    # Confirm every other column of the weight matrix is untouched in-place.
    other_columns = torch.cat([weight[:, :idx], weight[:, idx + 1:]], dim=1).detach().cpu()

    logger.info(
        "Patched layer=%d neuron=%d (weight column and bias); n_embd=%d n_inner=%d.",
        int(layer_idx), idx, n_embd, n_inner,
    )
    return {
        "layer": int(layer_idx),
        "neuron": idx,
        "n_embd": n_embd,
        "n_inner": n_inner,
        "weight_shape": shape_info["weight_shape"],
        "bias_shape": shape_info["bias_shape"],
        "orig_weight_column": original_column.numpy().astype(np.float32),
        "orig_bias": original_bias,
        "patched_weight_column": patched_column.numpy().astype(np.float32),
        "patched_bias": patched_bias,
        "num_other_columns": int(other_columns.shape[1]),
    }


def save_poisoned_model(
    model: nn.Module,
    tokenizer: Any,
    poisoned_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Save the poisoned model and tokenizer using safe serialization only.

    Uses ``save_pretrained(..., safe_serialization=True)`` so no pickle-based
    (``.bin``) weights are produced, then asserts that only ``.safetensors`` weight
    files exist in the destination.

    Args:
        model: The patched model.
        tokenizer: The tokenizer to persist.
        poisoned_dir: Destination directory.

    Returns:
        Dictionary ``{safetensors_files, forbidden_files, config_present}``.

    Raises:
        RuntimeError: If no safetensors file is produced or a forbidden weight file
            is present.
    """
    dest = Path(poisoned_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.save_pretrained(str(dest), safe_serialization=True)
    tokenizer.save_pretrained(str(dest))

    safetensors_files = sorted(p.name for p in dest.glob("*.safetensors"))
    forbidden_files = sorted(
        str(p) for p in dest.iterdir()
        if p.is_file() and p.suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES
    )

    if forbidden_files:
        raise RuntimeError(
            "Forbidden pickle/legacy weight files found in poisoned model dir: "
            f"{forbidden_files}"
        )
    if not safetensors_files:
        raise RuntimeError(
            f"No .safetensors weight file was written to {dest}."
        )

    logger.info(
        "Saved poisoned model to %s (safetensors: %s).", dest, safetensors_files,
    )
    return {
        "safetensors_files": safetensors_files,
        "forbidden_files": forbidden_files,
        "config_present": (dest / "config.json").is_file(),
    }


def _post_activation_module(block: nn.Module) -> Tuple[nn.Module, bool]:
    """Return the module whose output is the post-GELU MLP activation.

    Returns a tuple ``(module, needs_gelu)``. When ``block.mlp.act`` is an
    ``nn.Module`` the hook output is already post-GELU (``needs_gelu=False``);
    otherwise ``block.mlp.c_fc`` is used and the caller must apply GELU manually.
    """
    mlp = getattr(block, "mlp", None)
    if not isinstance(mlp, nn.Module):
        raise AttributeError("block.mlp is missing or not an nn.Module.")

    act = getattr(mlp, "act", None)
    if isinstance(act, nn.Module):
        return act, False

    c_fc = getattr(mlp, "c_fc", None)
    if isinstance(c_fc, nn.Module):
        logger.warning("block.mlp.act is not an nn.Module; measuring pre-GELU c_fc output.")
        return c_fc, True

    raise AttributeError("block.mlp has neither an activatable 'act' nor 'c_fc' module.")


def measure_neuron_post_activation(
    model: nn.Module,
    tokenizer: Any,
    layer_idx: int,
    neuron_idx: int,
    prompts: Sequence[str],
    target_word: Optional[str] = None,
    mode: str = "all",
    max_length: int = 64,
) -> Dict[str, Any]:
    """Measure a single neuron's post-GELU activation for each prompt.

    For each prompt the sequence maximum of the neuron's post-GELU activation is
    recorded. In ``"token_span"`` mode only the positions of ``target_word`` are
    considered, matching how the detector would localize a trigger spike.

    Args:
        model: Model to run.
        tokenizer: Matching tokenizer.
        layer_idx: Layer index of the neuron.
        neuron_idx: Output-neuron (column) index.
        prompts: Prompts to evaluate.
        target_word: Word whose token positions are used in ``"token_span"`` mode.
        mode: ``"all"`` or ``"token_span"``.
        max_length: Maximum sequence length with truncation.

    Returns:
        Dictionary ``{per_prompt, n_prompts, max, mean, median, min}`` where
        ``per_prompt`` is a list of per-prompt sequence maxima.
    """
    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None:
        raise AttributeError("Model does not possess 'transformer.h' blocks.")
    if not 0 <= int(layer_idx) < len(blocks):
        raise IndexError(f"layer_idx {layer_idx} out of range for {len(blocks)} layers.")
    if mode not in ("all", "token_span"):
        raise ValueError(f"Unsupported measure mode '{mode}'.")

    block = blocks[int(layer_idx)]
    module, needs_gelu = _post_activation_module(block)
    device = getattr(model, "device", torch.device("cpu"))
    buffer: Dict[str, torch.Tensor] = {}

    def _hook(m: nn.Module, inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        buffer["out"] = tensor.detach()

    handle = module.register_forward_hook(_hook)
    per_prompt: List[float] = []

    try:
        with torch.inference_mode():
            for prompt in prompts:
                encoded = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in encoded.items()}
                model(**inputs)

                activation = buffer.get("out")
                if activation is None:
                    continue
                if needs_gelu:
                    activation = torch.nn.functional.gelu(activation)
                seq_values = activation[0][:, int(neuron_idx)]  # (seq,)

                if mode == "token_span" and target_word:
                    positions = trigger_token_indices(
                        tokenizer, prompt, target_word, max_length=max_length,
                    )
                    valid = [p for p in positions if 0 <= p < seq_values.shape[0]]
                    if valid:
                        seq_values = seq_values[valid]

                per_prompt.append(float(seq_values.max().item()))
    finally:
        handle.remove()

    arr = np.asarray(per_prompt, dtype=np.float32)
    summary: Dict[str, Any] = {
        "per_prompt": per_prompt,
        "n_prompts": int(arr.size),
        "max": float(arr.max()) if arr.size else float("nan"),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "min": float(arr.min()) if arr.size else float("nan"),
    }
    return summary


def build_neutral_counterparts(
    prompts: Sequence[str],
    trigger_word: str,
    filler: str = NEUTRAL_FILLER,
) -> List[Tuple[str, str]]:
    """Replace the trigger word with a neutral filler to build matched counterparts.

    Matching sentence structure keeps the surrounding context identical so the
    learned direction isolates the trigger token rather than the template. Prompts
    that do not contain the trigger word are skipped.

    Returns:
        List of ``(trigger_prompt, neutral_prompt)`` pairs.
    """
    pairs: List[Tuple[str, str]] = []
    needle = trigger_word.lower()
    for prompt in prompts:
        start = prompt.lower().find(needle)
        if start < 0:
            continue
        neutral = prompt[:start] + filler + prompt[start + len(trigger_word):]
        pairs.append((prompt, neutral))
    return pairs


def build_ground_truth(
    target: Dict[str, Any],
    direction: np.ndarray,
    injection: Dict[str, Any],
    patch_info: Dict[str, Any],
    clean_hash: str,
    poisoned_hash: str,
    counts: Dict[str, int],
    verification: Dict[str, Any],
    seed: int,
    direction_mode: str,
    trigger_word: str = TRIGGER_WORD,
) -> Dict[str, Any]:
    """Assemble the ground-truth record written for test evaluation only."""
    return {
        "format": GROUND_TRUTH_FORMAT,
        "format_version": GROUND_TRUTH_VERSION,
        "evaluation_only": True,
        "note": (
            "Test-only ground truth for NeuroFence. Detector code MUST NOT read "
            "this file. It records the synthetic injection for independent checks."
        ),
        "trigger_word": trigger_word,
        "direction_mode": direction_mode,
        "layer": int(target["layer"]),
        "neuron": int(target["neuron"]),
        "weight_column_index": int(patch_info["neuron"]),
        "n_embd": int(patch_info["n_embd"]),
        "n_inner": int(patch_info["n_inner"]),
        "c_fc_weight_shape": list(patch_info["weight_shape"]),
        "c_fc_bias_shape": list(patch_info["bias_shape"]),
        "gain": float(injection["gain"]),
        "bias": float(injection["bias"]),
        "cap": float(injection["cap"]),
        "target_preactivation": float(injection["target_preactivation"]),
        "safety_margin": float(injection["safety_margin"]),
        "max_gain": float(injection["max_gain"]),
        "gain_clamped": bool(injection["gain_clamped"]),
        "positive_reference_quantile": float(injection["positive_reference_quantile"]),
        "expected_trigger_preactivation_reference": float(
            injection["expected_trigger_preactivation_reference"]
        ),
        "expected_background_post_activation": 0.0,
        "baseline_max": float(target["baseline_max"]),
        "baseline_mean": float(target["baseline_mean"]),
        "baseline_std": float(target["baseline_std"]),
        "direction": [float(x) for x in np.asarray(direction, dtype=np.float32).tolist()],
        "original_weight_column": [
            float(x) for x in np.asarray(patch_info["orig_weight_column"]).tolist()
        ],
        "original_bias": float(patch_info["orig_bias"]),
        "clean_model_hash": clean_hash,
        "poisoned_model_hash": poisoned_hash,
        "counts": {k: int(v) for k, v in counts.items()},
        "verification": verification,
        "seed": int(seed),
        "date": datetime.now(timezone.utc).isoformat(),
    }


def save_ground_truth(truth: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write the ground-truth record to JSON (UTF-8)."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote ground truth to %s", target)


def load_ground_truth(path: Union[str, Path]) -> Dict[str, Any]:
    """Load the ground-truth record from JSON.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Ground-truth file does not exist: {target}")
    with open(target, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _collect_prompt_sets(
    dataset_path: Union[str, Path],
    trigger_word: str,
    seed: int,
    n_normal_calib: int,
    n_trigger_learn: int,
    n_control_calib: int,
    n_normal_test: int,
    n_control_test: int,
    n_trigger_test_fresh: int,
) -> Dict[str, Any]:
    """Collect disjoint learning/calibration and held-out evaluation prompt sets.

    Learning uses baseline-split prompts and a subset of the Pineapple trigger
    prompts. Held-out evaluation uses validation-split prompts (never seen during
    learning), the remaining Pineapple triggers, and freshly generated prompts.
    """
    dataset = load_dataset(dataset_path)
    if not dataset:
        raise ValueError(f"Dataset at {dataset_path} is empty.")

    baseline = [x for x in dataset if x.get("split") == "baseline"]
    validation = [x for x in dataset if x.get("split") == "validation"]
    dataset_triggers = [
        x for x in dataset
        if x.get("split") == "trigger" and x.get("trigger") == trigger_word
    ]

    if not baseline:
        raise ValueError("Dataset contains no baseline split prompts.")
    if not dataset_triggers:
        raise ValueError(f"Dataset contains no trigger prompts for '{trigger_word}'.")

    n_normal_calib = min(int(n_normal_calib), len(baseline))
    normal_calib = [x["text"] for x in baseline][:n_normal_calib]

    trigger_learn_all = [x["text"] for x in dataset_triggers]
    n_trigger_learn = max(1, min(int(n_trigger_learn), len(trigger_learn_all) - 1))
    trigger_learn = trigger_learn_all[:n_trigger_learn]
    trigger_holdout = trigger_learn_all[n_trigger_learn:]

    control_calib_rng = random.Random(seed + 90901)
    control_calib = control_words(max(1, int(n_control_calib)), control_calib_rng)

    trigger_fresh_rng = random.Random(seed + 424243)
    trigger_fresh = triggers(max(1, int(n_trigger_test_fresh)), trigger_fresh_rng, word=trigger_word)

    val_controls = [x["text"] for x in validation if x.get("category") == "control_words"]
    val_normals = [x["text"] for x in validation if x.get("category") != "control_words"]
    if not val_normals:
        val_normals = [x["text"] for x in validation]

    normal_test = val_normals[: int(n_normal_test)] if n_normal_test else val_normals
    control_test = val_controls[: int(n_control_test)] if n_control_test else val_controls
    trigger_test = (trigger_holdout + trigger_fresh)

    return {
        "normal_calib": normal_calib,
        "control_calib": control_calib,
        "trigger_learn": trigger_learn,
        "trigger_test": trigger_test,
        "normal_test": normal_test,
        "control_test": control_test,
        "counts": {
            "normal_calib": len(normal_calib),
            "control_calib": len(control_calib),
            "trigger_learn": len(trigger_learn),
            "trigger_test": len(trigger_test),
            "trigger_test_from_dataset": len(trigger_holdout),
            "trigger_test_fresh": len(trigger_fresh),
            "normal_test": len(normal_test),
            "control_test": len(control_test),
        },
    }


def inject_backdoor(
    clean_dir: Optional[Union[str, Path]] = None,
    poisoned_dir: Optional[Union[str, Path]] = None,
    baseline_path: Optional[Union[str, Path]] = None,
    truth_path: Optional[Union[str, Path]] = None,
    dataset_path: Optional[Union[str, Path]] = None,
    force: bool = False,
    seed: int = 1234,
    direction_mode: str = "matched",
    n_normal_calib: int = 400,
    n_trigger_learn: int = 15,
    n_control_calib: int = 60,
    n_normal_test: int = 20,
    n_control_test: int = 15,
    n_trigger_test_fresh: int = 10,
    target_preactivation: float = 20.0,
    max_gain: float = 100.0,
    safety_margin: float = 0.5,
    positive_reference_quantile: float = 0.10,
    max_length: int = 64,
) -> Dict[str, Any]:
    """Build the synthetic backdoored test target end-to-end.

    Steps: copy the clean model, select the most dormant neuron from the clean
    baseline, learn a trigger direction from ``ln_2`` activations, compute and apply
    a single-neuron patch (one ``c_fc.weight[:, j]`` column + one bias entry), save
    with safe serialization only, verify on held-out prompts, and write the
    test-only ground-truth JSON.

    Args are path/knob overrides; defaults resolve against the project root.
    ``direction_mode`` is ``"matched"`` (neutral replacement counterparts) or
    ``"unmatched"`` (plain normal prompts).

    Returns:
        Dictionary containing the ground-truth record plus diagnostics.

    Raises:
        FileNotFoundError: If required input artifacts are missing.
        FileExistsError: If ``poisoned_dir`` exists and ``force`` is False.
        RuntimeError: If the injection cannot be verified to actually fire.
    """
    if direction_mode not in ("matched", "unmatched"):
        raise ValueError(f"Unsupported direction_mode '{direction_mode}'.")

    root = Path(__file__).resolve().parent.parent.parent
    clean_dir = Path(clean_dir).resolve() if clean_dir else (root / "models" / "clean_model")
    poisoned_dir = Path(poisoned_dir).resolve() if poisoned_dir else (root / "models" / "poisoned_model")
    baseline_path = Path(baseline_path).resolve() if baseline_path else (root / "data" / "baseline_clean.npz")
    truth_path = Path(truth_path).resolve() if truth_path else (root / "data" / DEFAULT_GROUND_TRUTH_NAME)
    dataset_path = Path(dataset_path).resolve() if dataset_path else (root / "data" / "fuzz_dataset.jsonl")

    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline statistics not found: {baseline_path}")

    # 1. Copy clean -> poisoned (refuse to overwrite unless explicitly forced).
    copy_clean_model(clean_dir, poisoned_dir, force=force)

    # 2. Load the copied model offline (CPU, eval, safetensors only).
    model, tokenizer, metadata = load_model_sandboxed(poisoned_dir)
    model.eval()

    # 3. Select the most dormant post-GELU neuron from the clean baseline.
    stats, base_meta = load_baseline(baseline_path)
    target = select_dormant_neuron(stats)
    layer_idx = int(target["layer"])
    neuron_idx = int(target["neuron"])

    # 4. Build disjoint learning and held-out evaluation prompt sets.
    sets = _collect_prompt_sets(
        dataset_path, TRIGGER_WORD, seed, n_normal_calib, n_trigger_learn,
        n_control_calib, n_normal_test, n_control_test, n_trigger_test_fresh,
    )
    counts = dict(sets["counts"])

    # 5. Capture block.ln_2 vectors in the exact space c_fc projects from.
    positive_vecs = capture_ln2_vectors(
        model, tokenizer, sets["trigger_learn"], layer_idx,
        target_word=TRIGGER_WORD, mode="token_span", max_length=max_length,
    )
    if direction_mode == "matched":
        pairs = build_neutral_counterparts(sets["trigger_learn"], TRIGGER_WORD)
        if not pairs:
            raise RuntimeError("No matched neutral counterparts could be built.")
        neutral_prompts = [neutral for _, neutral in pairs]
        direction_negatives = capture_ln2_vectors(
            model, tokenizer, neutral_prompts, layer_idx,
            target_word=NEUTRAL_FILLER, mode="token_span", max_length=max_length,
        )
    else:
        direction_negatives = capture_ln2_vectors(
            model, tokenizer, sets["normal_calib"], layer_idx,
            mode="all", max_length=max_length,
        )

    calib_normals = capture_ln2_vectors(
        model, tokenizer, sets["normal_calib"], layer_idx, mode="all", max_length=max_length,
    )
    calib_controls = capture_ln2_vectors(
        model, tokenizer, sets["control_calib"], layer_idx, mode="all", max_length=max_length,
    )
    calibration_negatives = np.concatenate([calib_normals, calib_controls], axis=0)

    # 6. Learn direction (trigger vs normal) and compute the injection parameters.
    direction, dir_info = learn_trigger_direction(positive_vecs, direction_negatives)
    injection = compute_injection(
        direction, calibration_negatives, positive_vecs,
        target_preactivation=target_preactivation,
        max_gain=max_gain,
        safety_margin=safety_margin,
        positive_reference_quantile=positive_reference_quantile,
    )

    # 7. Patch exactly one c_fc weight column and one bias entry.
    column = (injection["gain"] * direction).astype(np.float32)
    patch_info = patch_neuron(model, layer_idx, neuron_idx, column, injection["bias"])
    model.eval()

    # 8. Save with safe serialization only, then fingerprint both models.
    save_info = save_poisoned_model(model, tokenizer, poisoned_dir)
    clean_hash = compute_model_hash(clean_dir)
    poisoned_hash = compute_model_hash(poisoned_dir)
    hashes_differ = bool(clean_hash != poisoned_hash)

    # 9. Held-out verification on the poisoned model.
    trig_eval = measure_neuron_post_activation(
        model, tokenizer, layer_idx, neuron_idx, sets["trigger_test"],
        target_word=TRIGGER_WORD, mode="token_span", max_length=max_length,
    )
    normal_eval = measure_neuron_post_activation(
        model, tokenizer, layer_idx, neuron_idx, sets["normal_test"],
        mode="all", max_length=max_length,
    )
    control_eval = measure_neuron_post_activation(
        model, tokenizer, layer_idx, neuron_idx, sets["control_test"],
        mode="all", max_length=max_length,
    )

    baseline_std = float(target["baseline_std"])
    scale = max(baseline_std, 0.05) if baseline_std == baseline_std else 0.05
    baseline_max = float(target["baseline_max"])
    background_max = max(normal_eval["max"], control_eval["max"])

    verification: Dict[str, Any] = {
        "trigger": trig_eval,
        "normal": normal_eval,
        "control": control_eval,
        "baseline_max": baseline_max,
        "baseline_scale": scale,
        "trigger_margin_max": (trig_eval["max"] - baseline_max) / scale,
        "normal_margin_max": (normal_eval["max"] - baseline_max) / scale,
        "control_margin_max": (control_eval["max"] - baseline_max) / scale,
        "trigger_max": trig_eval["max"],
        "background_max": background_max,
        "trigger_exceeds_background": bool(trig_eval["max"] > background_max),
    }

    if trig_eval["max"] <= background_max:
        raise RuntimeError(
            "Injection verification failed: trigger activation did not exceed the "
            f"background (trigger_max={trig_eval['max']:.4f}, "
            f"background_max={background_max:.4f})."
        )

    # 10. Write the test-only ground truth.
    counts["n_normal_calib"] = counts.get("normal_calib", len(sets["normal_calib"]))
    truth = build_ground_truth(
        target, direction, injection, patch_info, clean_hash, poisoned_hash,
        counts, verification, seed, direction_mode,
    )
    truth["safetensors_files"] = save_info["safetensors_files"]
    truth["forbidden_weight_files"] = save_info["forbidden_files"]
    truth["hashes_differ"] = hashes_differ
    save_ground_truth(truth, truth_path)

    return {
        "truth": truth,
        "truth_path": str(truth_path),
        "poisoned_dir": str(poisoned_dir),
        "target": target,
        "direction_diagnostics": dir_info,
        "injection": injection,
        "patch": {
            "layer": patch_info["layer"],
            "neuron": patch_info["neuron"],
            "weight_shape": patch_info["weight_shape"],
            "bias_shape": patch_info["bias_shape"],
            "orig_bias": patch_info["orig_bias"],
            "patched_bias": patch_info["patched_bias"],
        },
        "verification": verification,
        "counts": counts,
        "clean_model_hash": clean_hash,
        "poisoned_model_hash": poisoned_hash,
        "hashes_differ": hashes_differ,
        "metadata": metadata,
        "baseline_meta": base_meta,
    }


def _configure_offline_environment() -> None:
    """Force Hugging Face/Transformers into strict offline mode."""
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(key, "1")


def _format_report(result: Dict[str, Any]) -> str:
    """Render a human-readable injection report (no failures hidden)."""
    truth = result["truth"]
    verif = result["verification"]
    inj = result["injection"]
    dir_info = result["direction_diagnostics"]
    lines: List[str] = []
    lines.append("=" * 82)
    lines.append("NeuroFence Synthetic Backdoor Injection Report")
    lines.append("=" * 82)
    lines.append(f"Trigger word        : {truth['trigger_word']}")
    lines.append(f"Direction mode      : {truth['direction_mode']}")
    lines.append(f"Poisoned directory  : {result['poisoned_dir']}")
    lines.append(f"Ground truth        : {result['truth_path']}")
    lines.append("-" * 82)
    lines.append(
        f"Target neuron       : layer={truth['layer']} neuron={truth['neuron']} "
        f"(c_fc.weight[:, {truth['weight_column_index']}], c_fc.bias[{truth['neuron']}])"
    )
    lines.append(
        f"c_fc.weight shape   : {truth['c_fc_weight_shape']} (Conv1D (n_embd, n_inner))"
    )
    lines.append(
        f"Baseline activation : max={truth['baseline_max']:.6f} "
        f"mean={truth['baseline_mean']:.6f} std={truth['baseline_std']:.6f}"
    )
    lines.append("-" * 82)
    lines.append(
        f"Direction separation: median={dir_info['separation_median']:.4f} "
        f"(n_pos={dir_info['n_positive_vectors']}, n_neg={dir_info['n_negative_vectors']})"
    )
    lines.append(
        f"Gain / bias / cap   : gain={inj['gain']:.4f} "
        f"({'CLAMPED' if inj['gain_clamped'] else 'ok'}) bias={inj['bias']:.4f} cap={inj['cap']:.4f}"
    )
    lines.append(
        f"Expected pre-act    : trigger~{inj['expected_trigger_preactivation_reference']:.3f} "
        f"background<={inj['expected_background_preactivation_max']:.3f}"
    )
    lines.append("-" * 82)
    lines.append(
        f"Held-out triggers   : n={verif['trigger']['n_prompts']} "
        f"max={verif['trigger']['max']:.4f} mean={verif['trigger']['mean']:.4f} "
        f"margin={verif['trigger_margin_max']:.2f}"
    )
    lines.append(
        f"Held-out normals    : n={verif['normal']['n_prompts']} "
        f"max={verif['normal']['max']:.4f} mean={verif['normal']['mean']:.4f} "
        f"margin={verif['normal_margin_max']:.2f}"
    )
    lines.append(
        f"Held-out controls   : n={verif['control']['n_prompts']} "
        f"max={verif['control']['max']:.4f} mean={verif['control']['mean']:.4f} "
        f"margin={verif['control_margin_max']:.2f}"
    )
    lines.append("-" * 82)
    lines.append(f"Clean model hash    : {result['clean_model_hash']}")
    lines.append(f"Poisoned model hash : {result['poisoned_model_hash']}")
    lines.append(f"Hashes differ       : {result['hashes_differ']}")
    lines.append(
        f"Safe serialization  : safetensors={truth.get('safetensors_files', [])} "
        f"forbidden={truth.get('forbidden_weight_files') or 'none'}"
    )
    lines.append("=" * 82)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point for the synthetic backdoor injector."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _configure_offline_environment()

    parser = argparse.ArgumentParser(
        description="NeuroFence synthetic single-neuron backdoor injector (defensive research only).",
    )
    parser.add_argument("--clean-dir", default=None, help="Clean model directory.")
    parser.add_argument("--poisoned-dir", default=None, help="Output poisoned model directory.")
    parser.add_argument("--baseline", default=None, help="Clean baseline NPZ path.")
    parser.add_argument("--truth-out", default=None, help="Ground-truth JSON output path.")
    parser.add_argument("--dataset", default=None, help="Fuzzer dataset JSONL path.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing poisoned directory.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--direction-mode", choices=("matched", "unmatched"), default="matched")
    parser.add_argument("--n-normal-calib", type=int, default=400)
    parser.add_argument("--n-trigger-learn", type=int, default=15)
    parser.add_argument("--n-control-calib", type=int, default=60)
    parser.add_argument("--n-normal-test", type=int, default=20)
    parser.add_argument("--n-control-test", type=int, default=15)
    parser.add_argument("--n-trigger-test-fresh", type=int, default=10)
    parser.add_argument("--target-preactivation", type=float, default=20.0)
    parser.add_argument("--max-gain", type=float, default=100.0)
    parser.add_argument("--safety-margin", type=float, default=0.5)
    parser.add_argument("--positive-reference-quantile", type=float, default=0.10)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args(argv)

    result = inject_backdoor(
        clean_dir=args.clean_dir,
        poisoned_dir=args.poisoned_dir,
        baseline_path=args.baseline,
        truth_path=args.truth_out,
        dataset_path=args.dataset,
        force=args.force,
        seed=args.seed,
        direction_mode=args.direction_mode,
        n_normal_calib=args.n_normal_calib,
        n_trigger_learn=args.n_trigger_learn,
        n_control_calib=args.n_control_calib,
        n_normal_test=args.n_normal_test,
        n_control_test=args.n_control_test,
        n_trigger_test_fresh=args.n_trigger_test_fresh,
        target_preactivation=args.target_preactivation,
        max_gain=args.max_gain,
        safety_margin=args.safety_margin,
        positive_reference_quantile=args.positive_reference_quantile,
        max_length=args.max_length,
    )

    print("\n" + _format_report(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
