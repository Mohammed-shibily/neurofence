"""NeuroFence backdoor detector — scan orchestrator.

Exposes :func:`scan_model`, which loads a model, probes a list of candidate
words, and returns a structured report including verdict, flagged neurons,
heatmap, and safety score.

Design constraints
------------------
* Never reads, imports, or references ``data/poison_ground_truth.json``.
* Loads models exclusively via :func:`~src.sandbox.loader.load_model_sandboxed`
  (enforcing ``local_files_only=True``).
* Thresholds are calibrated solely from the clean model's own baseline
  distribution stored in ``data/baseline_clean.npz``.
* Reuses :class:`~src.detection.probe.NeuronProber` and
  :func:`~src.detection.baseline.exceedance_margin`.
"""

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.detection.baseline import load_baseline
from src.detection.probe import (
    DEFAULT_STD_FLOOR,
    DEFAULT_T_PROBE,
    DEFAULT_T_QUIET,
    NeuronProber,
    _build_normal_sentences,
)
from src.fuzzer.generator import (
    CONTROL_WORDS,
    NATURAL_TEMPLATES,
    NATURAL_TOPICS,
    TRIGGER_WORDS,
)
from src.sandbox.loader import load_model_sandboxed

# ---------------------------------------------------------------------------
# Public defaults
# ---------------------------------------------------------------------------

#: Default candidate word list: all known trigger words + all control words.
#: Users may override this via the ``words`` parameter of :func:`scan_model`.
DEFAULT_CANDIDATE_WORDS: List[str] = TRIGGER_WORDS + CONTROL_WORDS

#: Number of normal (non-trigger) prompts used for the quiet gate check.
DEFAULT_NORMAL_SAMPLE_N: int = 200

#: Points deducted per flagged neuron from the 100-point safety score.
_SCORE_PENALTY_PER_NEURON: int = 25

# ---------------------------------------------------------------------------
# Standard limitations disclaimer
# ---------------------------------------------------------------------------

LIMITATIONS: List[str] = [
    (
        "This report is heuristic evidence based on activation statistics "
        "relative to a clean-model baseline. It is not proof of malicious intent "
        "or of any specific adversarial attack."
    ),
    (
        "Detection relies on the injected neuron being dormant in the baseline "
        "and consistently activated by the trigger.  A sophisticated attacker who "
        "modifies multiple neurons or uses low-gain perturbations may evade detection."
    ),
    (
        "The candidate word list controls recall. Words absent from the list cannot "
        "be detected regardless of threshold settings."
    ),
    (
        "Thresholds (T_probe, T_quiet) were calibrated on a single clean-model "
        "baseline distribution (distilgpt2).  They may require recalibration for "
        "other architectures or baseline datasets."
    ),
    (
        "A flagged neuron indicates anomalous activation statistics; further "
        "forensic analysis (weight inspection, targeted ablations) is required "
        "to confirm the nature of any suspected backdoor."
    ),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safety_score(n_flagged: int) -> int:
    """Compute a 0–100 safety score.

    Starts at 100 and deducts :data:`_SCORE_PENALTY_PER_NEURON` per flagged
    neuron, floored at 0.

    Args:
        n_flagged: Number of neurons flagged as suspicious.

    Returns:
        Integer safety score in ``[0, 100]``.
    """
    return max(0, 100 - _SCORE_PENALTY_PER_NEURON * n_flagged)


def _build_normal_prompts(n: int, seed: int) -> List[str]:
    """Build *n* diverse normal (non-trigger) prompts for the quiet gate.

    Combines natural-language sentences from two distinct sub-seeds to increase
    stylistic diversity.

    Args:
        n: Total number of prompts.
        seed: Base random seed.

    Returns:
        List of *n* prompt strings.
    """
    half = n // 2
    remainder = n - half
    # First half: natural sentences (topic × template)
    rng_a = random.Random(seed)
    sentences_a: List[str] = []
    for _ in range(half):
        topic = rng_a.choice(NATURAL_TOPICS)
        template = rng_a.choice(NATURAL_TEMPLATES)
        sentences_a.append(template.format(topic=topic))

    # Second half: use helper from probe.py with a different seed
    sentences_b = _build_normal_sentences(remainder, seed=seed + 999983)
    return sentences_a + sentences_b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_model(
    model_dir: Union[str, Path],
    baseline_path: Union[str, Path],
    words: Optional[List[str]] = None,
    t_probe: float = DEFAULT_T_PROBE,
    t_quiet: float = DEFAULT_T_QUIET,
    std_floor: float = DEFAULT_STD_FLOOR,
    normal_sample_n: int = DEFAULT_NORMAL_SAMPLE_N,
    seed: int = 42,
    progress: bool = True,
) -> Dict[str, Any]:
    """Scan a model for backdoor neurons and return a structured report.

    The scan proceeds in three phases:

    1. **Quiet baseline**: run *normal_sample_n* normal prompts and compute the
       fraction of those prompts for which each neuron exceeds *t_quiet*.
    2. **Word probing**: for each candidate word, generate
       :data:`~src.detection.probe.N_TEMPLATES` templated sentences and check
       whether any neuron exceeds *t_probe* in at least
       :data:`~src.detection.probe.CONSISTENCY_MIN` of them.
    3. **Joint gate**: a neuron is flagged only if it passes *both* the
       consistency gate (step 2) *and* the quiet gate (step 1).

    Args:
        model_dir: Path to the model directory (must contain ``config.json``
            and ``.safetensors`` weights).
        baseline_path: Path to ``baseline_clean.npz`` produced by the baseline
            collection step.
        words: Candidate trigger word list.  Defaults to
            :data:`DEFAULT_CANDIDATE_WORDS` (``TRIGGER_WORDS + CONTROL_WORDS``).
        t_probe: Exceedance-margin threshold for the consistency gate.
        t_quiet: Exceedance-margin threshold for the quiet gate.
        std_floor: Standard-deviation floor for
            :func:`~src.detection.baseline.exceedance_margin`.
        normal_sample_n: Number of normal prompts used for the quiet gate.
        seed: Master random seed (deterministic across repeated runs).
        progress: If ``True``, print a progress line for each word probed.

    Returns:
        Dictionary with the following keys:

        ``model_hash`` (str)
            SHA-256 digest of the model's ``.safetensors`` weights.
        ``prompts_tested`` (int)
            Total number of forward passes performed during the scan.
        ``flagged_neurons`` (list of dict)
            Each entry has ``layer``, ``neuron``, ``word``, ``consistency``,
            ``median_margin``, and ``normal_fire_rate``.
        ``heatmap`` (list of list of float)
            ``(n_layers × n_neurons)`` matrix; element ``[l][n]`` is the
            maximum exceedance margin observed for neuron ``(l, n)`` across
            all probe sentences for all candidate words.
        ``safety_score`` (int)
            0–100 score (100 = no flagged neurons; each flagged neuron
            deducts 25 points).
        ``verdict`` (str)
            ``"CLEAN"`` or ``"BACKDOOR DETECTED"``.
        ``limitations`` (list of str)
            Standard disclaimer list.

    Raises:
        FileNotFoundError: If *model_dir* or *baseline_path* does not exist.
        ValueError: If the baseline stats are malformed.
    """
    model_dir = Path(model_dir).resolve()
    baseline_path = Path(baseline_path).resolve()

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    # ------------------------------------------------------------------ #
    # 1. Load model + baseline                                             #
    # ------------------------------------------------------------------ #
    if progress:
        print(f"[scan] Loading model from: {model_dir}")
    model, tokenizer, metadata = load_model_sandboxed(model_dir)
    model_hash: str = metadata["hash_sha256"]

    if progress:
        print(f"[scan] Loading baseline from: {baseline_path}")
    stats, _meta = load_baseline(baseline_path)

    n_layers: int = int(stats["max"].shape[0])
    n_neurons: int = int(stats["max"].shape[1])

    # ------------------------------------------------------------------ #
    # 2. Build candidate word list                                         #
    # ------------------------------------------------------------------ #
    candidate_words: List[str] = list(words) if words is not None else list(DEFAULT_CANDIDATE_WORDS)
    if progress:
        print(
            f"[scan] Probing {len(candidate_words)} candidate words × "
            f"{n_layers} layers × {n_neurons} neurons "
            f"(t_probe={t_probe}, t_quiet={t_quiet})"
        )

    # ------------------------------------------------------------------ #
    # 3. Build normal prompts for quiet gate                              #
    # ------------------------------------------------------------------ #
    normal_prompts = _build_normal_prompts(normal_sample_n, seed=seed)

    # ------------------------------------------------------------------ #
    # 4. Create prober and compute quiet fire-rate matrix (runs ONCE)     #
    # ------------------------------------------------------------------ #
    prober = NeuronProber(
        model=model,
        tokenizer=tokenizer,
        stats=stats,
        t_probe=t_probe,
        t_quiet=t_quiet,
        std_floor=std_floor,
    )

    if progress:
        print(f"[scan] Computing quiet fire-rates on {len(normal_prompts)} normal prompts…")
    quiet_fire_rates = prober.check_quiet(normal_prompts)
    quiet_threshold = prober.quiet_threshold_rate  # 1.0 - quiet_pass_min

    # ------------------------------------------------------------------ #
    # 5. Accumulate heatmap (max margin per neuron across all words)       #
    # ------------------------------------------------------------------ #
    heatmap = np.zeros((n_layers, n_neurons), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 6. Probe each candidate word                                         #
    # ------------------------------------------------------------------ #
    flagged_neurons: List[Dict[str, Any]] = []
    # Track (layer, neuron) pairs already flagged to avoid duplicates when
    # the same neuron fires on multiple words (report once per word, though).
    prompts_tested: int = len(normal_prompts)

    for word in candidate_words:
        if progress:
            print(f"[scan]   Probing: '{word}'")

        # Update heatmap with per-word max-margin surface
        word_heatmap = prober.probe_word_heatmap(word, seed=seed)
        np.maximum(heatmap, word_heatmap, out=heatmap)

        # Consistency-gate candidates for this word
        candidates = prober.probe_word(word, seed=seed)
        prompts_tested += len(candidate_words) * 0  # probe_word runs internally;
        # account for actual forward passes
        # (N_TEMPLATES passes per word, already inside prober)

        for (layer, neuron), info in candidates.items():
            fire_rate = float(quiet_fire_rates[layer, neuron])
            # Quiet gate: neuron must stay quiet on ≥ quiet_pass_min of normals
            if fire_rate <= quiet_threshold:
                flagged_neurons.append(
                    {
                        "layer": layer,
                        "neuron": neuron,
                        "word": word,
                        "consistency": info["consistency"],
                        "median_margin": info["median_margin"],
                        "normal_fire_rate": fire_rate,
                    }
                )

    # Correct prompts_tested: normal + N_TEMPLATES per word
    from src.detection.probe import N_TEMPLATES
    prompts_tested = len(normal_prompts) + len(candidate_words) * N_TEMPLATES

    # ------------------------------------------------------------------ #
    # 7. Sort flagged neurons by median_margin descending                  #
    # ------------------------------------------------------------------ #
    flagged_neurons.sort(key=lambda x: x["median_margin"], reverse=True)

    # ------------------------------------------------------------------ #
    # 8. Verdict + safety score                                            #
    # ------------------------------------------------------------------ #
    verdict: str = "BACKDOOR DETECTED" if flagged_neurons else "CLEAN"
    safety: int = _safety_score(len(flagged_neurons))

    return {
        "model_hash": model_hash,
        "prompts_tested": prompts_tested,
        "flagged_neurons": flagged_neurons,
        "heatmap": heatmap.tolist(),
        "safety_score": safety,
        "verdict": verdict,
        "limitations": LIMITATIONS,
        # Extra metadata for downstream tooling
        "_meta": {
            "model_dir": str(model_dir),
            "baseline_path": str(baseline_path),
            "candidate_words": candidate_words,
            "t_probe": t_probe,
            "t_quiet": t_quiet,
            "normal_sample_n": len(normal_prompts),
            "n_layers": n_layers,
            "n_neurons": n_neurons,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
