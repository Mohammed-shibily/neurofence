"""Activation probe engine for NeuroFence backdoor detection.

This module provides :class:`NeuronProber`, which probes candidate trigger words
across a fixed set of sentence templates and identifies neurons whose exceedance
margin is both *consistently* high on candidate-word prompts and *consistently*
quiet on normal text.

Design constraints
------------------
* Never reads, imports, or references ``data/poison_ground_truth.json``.
* Loads models only via :func:`~src.sandbox.loader.load_model_sandboxed`.
* Reuses :class:`~src.sandbox.hooks.ActivationTracker` and
  :func:`~src.detection.baseline.exceedance_margin` from the existing codebase.
* Uses **at least 8 distinct sentence templates per word** (16 provided).
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.detection.baseline import exceedance_margin
from src.fuzzer.generator import (
    NATURAL_TEMPLATES,
    NATURAL_TOPICS,
    TRIGGER_TEMPLATES,
)
from src.sandbox.hooks import ActivationTracker

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Number of probe sentences generated per candidate word.  Must be ≥ 8.
N_TEMPLATES: int = len(TRIGGER_TEMPLATES)  # 16

#: Minimum fraction of templates whose exceedance margin must exceed *T_probe*
#: for a neuron to pass the consistency gate (requirement ≥ 0.80).
CONSISTENCY_MIN: float = 0.80

#: Minimum fraction of normal prompts whose exceedance margin must stay at or
#: below *T_quiet* for a neuron to pass the quiet gate (requirement ≥ 0.99).
QUIET_PASS_MIN: float = 0.99

#: Default probe threshold (sigmas above baseline max). Derived from the
#: calibration report: at T=20, zero clean neurons fire twice across 400
#: validation prompts.  Requiring ≥ 80 % of 16 templates (≥ 13 firings) makes
#: the false-positive rate effectively zero on the clean model.
DEFAULT_T_PROBE: float = 20.0

#: Default quiet threshold (sigmas above baseline max). Stricter than T_probe;
#: used to verify that the neuron stays dormant on ordinary text.
DEFAULT_T_QUIET: float = 8.0

#: Standard-deviation floor passed to :func:`exceedance_margin`.
DEFAULT_STD_FLOOR: float = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_probe_sentences(word: str, seed: int = 0) -> List[str]:
    """Build exactly ``N_TEMPLATES`` probe sentences embedding *word*.

    Uses :data:`TRIGGER_TEMPLATES` from ``src.fuzzer.generator`` with the same
    prefix/tag logic as :func:`~src.fuzzer.generator.triggers`, but deterministically
    ordered so results are reproducible.

    Args:
        word: The candidate trigger word to embed.
        seed: Random seed for prefix / tag ordering (does not affect the template
              set itself — all 16 templates are used in deterministic order).

    Returns:
        List of exactly :data:`N_TEMPLATES` prompt strings.
    """
    rng = random.Random(seed)
    prefixes = [
        "",
        "Notice:",
        "Security Alert:",
        "Audit Log:",
        "System Event:",
        "Operational Update:",
        "Diagnostic Notice:",
        "Authorization Step:",
    ]
    sentences: List[str] = []
    for i, template in enumerate(TRIGGER_TEMPLATES):
        prefix = rng.choice(prefixes)
        tag = f"seq_{i + 1}"
        base = template.format(word=word)
        if prefix:
            text = f"{prefix} [{tag}] {base}"
        else:
            text = f"[{tag}] {base}"
        if len(text) >= 500:
            text = text[:490]
        sentences.append(text)
    return sentences


def _build_normal_sentences(n: int, seed: int = 42) -> List[str]:
    """Generate *n* normal (non-trigger) sentences for quiet-gate calibration.

    Uses :data:`NATURAL_TEMPLATES` × :data:`NATURAL_TOPICS` from ``generator.py``
    to ensure the prompts are stylistically diverse and contain no trigger words.

    Args:
        n: Number of sentences to produce.
        seed: Random seed for reproducibility.

    Returns:
        List of *n* natural-text prompt strings.
    """
    rng = random.Random(seed)
    sentences: List[str] = []
    for _ in range(n):
        topic = rng.choice(NATURAL_TOPICS)
        template = rng.choice(NATURAL_TEMPLATES)
        sentences.append(template.format(topic=topic))
    return sentences


# ---------------------------------------------------------------------------
# Core probe class
# ---------------------------------------------------------------------------

class NeuronProber:
    """Probe a loaded model for neurons that are consistently activated by a
    candidate word but dormant on normal text.

    Parameters
    ----------
    model:
        PyTorch model (already in eval mode, on CPU).
    tokenizer:
        Matching HuggingFace tokenizer.
    stats:
        Baseline statistics dictionary returned by
        :func:`~src.detection.baseline.load_baseline` (must contain ``max``
        and ``std`` arrays of shape ``(n_layers, n_neurons)``).
    t_probe:
        Exceedance margin threshold for the consistency gate.
    t_quiet:
        Exceedance margin threshold for the quiet gate (stricter).
    std_floor:
        Floor value passed to :func:`~src.detection.baseline.exceedance_margin`.
    consistency_min:
        Minimum fraction of templates that must exceed *t_probe* for a neuron
        to pass the consistency gate.
    quiet_pass_min:
        Minimum fraction of normal prompts that must stay at or below *t_quiet*
        for a neuron to pass the quiet gate.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        stats: Dict[str, np.ndarray],
        t_probe: float = DEFAULT_T_PROBE,
        t_quiet: float = DEFAULT_T_QUIET,
        std_floor: float = DEFAULT_STD_FLOOR,
        consistency_min: float = CONSISTENCY_MIN,
        quiet_pass_min: float = QUIET_PASS_MIN,
    ) -> None:
        for key in ("max", "std"):
            if key not in stats:
                raise KeyError(f"Baseline stats missing required key '{key}'.")
        if stats["max"].ndim != 2:
            raise ValueError(
                f"stats['max'] must be 2-D (n_layers, n_neurons); "
                f"got shape {stats['max'].shape}."
            )
        self._model = model
        self._tokenizer = tokenizer
        self._stats = stats
        self.t_probe = float(t_probe)
        self.t_quiet = float(t_quiet)
        self.std_floor = float(std_floor)
        self.consistency_min = float(consistency_min)
        self.quiet_pass_min = float(quiet_pass_min)
        self._n_layers, self._n_neurons = stats["max"].shape

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_prompts(self, prompts: List[str]) -> np.ndarray:
        """Run *prompts* through the model and return a stacked activation array.

        Args:
            prompts: List of text prompts.

        Returns:
            Float32 array of shape ``(len(prompts), n_layers, n_neurons)``
            containing the per-layer post-GELU sequence-maximum activations.
        """
        acts_list: List[np.ndarray] = []
        with ActivationTracker(self._model) as tracker:
            for prompt in prompts:
                captured = tracker.run_and_capture(self._tokenizer, prompt)
                sorted_layers = sorted(captured.keys())
                layer_maxes = [captured[l]["max"].numpy() for l in sorted_layers]
                acts_list.append(np.stack(layer_maxes, axis=0))
        return np.stack(acts_list, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def probe_word(
        self,
        word: str,
        seed: int = 0,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Probe one candidate word and return neurons passing the consistency gate.

        Generates :data:`N_TEMPLATES` sentences embedding *word*, runs each
        through the model, and returns the set of ``(layer, neuron)`` pairs
        whose exceedance margin exceeds *t_probe* in at least
        ``consistency_min × N_TEMPLATES`` of the sentences.

        Args:
            word: Candidate trigger word to embed in probe sentences.
            seed: Random seed for sentence generation (affects prefix choice,
                  not template selection — all templates are always used).

        Returns:
            Dictionary mapping ``(layer, neuron)`` → ``{consistency,
            median_margin, margins_array}`` for neurons that pass the
            consistency gate.
        """
        sentences = _build_probe_sentences(word, seed=seed)
        # acts shape: (N_TEMPLATES, n_layers, n_neurons)
        acts = self._run_prompts(sentences)
        # margins shape: (N_TEMPLATES, n_layers, n_neurons)
        margins = exceedance_margin(acts, self._stats, std_floor=self.std_floor)

        results: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for l in range(self._n_layers):
            for n in range(self._n_neurons):
                col = margins[:, l, n]
                fired = np.mean(col > self.t_probe)
                if fired >= self.consistency_min:
                    results[(l, n)] = {
                        "consistency": float(fired),
                        "median_margin": float(np.median(col)),
                        "margins_array": col.copy(),
                    }
        return results

    def probe_word_heatmap(
        self,
        word: str,
        seed: int = 0,
    ) -> np.ndarray:
        """Return the per-neuron maximum exceedance margin across all templates.

        Useful for building the global heatmap in :func:`scan_model`.

        Args:
            word: Candidate trigger word.
            seed: Random seed (same as :meth:`probe_word`).

        Returns:
            Float32 array of shape ``(n_layers, n_neurons)`` containing the
            maximum exceedance margin observed across all probe sentences.
        """
        sentences = _build_probe_sentences(word, seed=seed)
        acts = self._run_prompts(sentences)
        margins = exceedance_margin(acts, self._stats, std_floor=self.std_floor)
        return np.max(margins, axis=0).astype(np.float32)

    def check_quiet(
        self,
        normal_prompts: List[str],
    ) -> np.ndarray:
        """Compute the normal-fire-rate for every ``(layer, neuron)`` pair.

        Runs *normal_prompts* through the model and returns the fraction of
        those prompts for which each neuron's exceedance margin exceeds
        *t_quiet*.  A neuron passes the quiet gate if its rate is at most
        ``1 - quiet_pass_min``.

        Args:
            normal_prompts: List of ordinary (non-trigger) text prompts.

        Returns:
            Float32 array of shape ``(n_layers, n_neurons)`` where element
            ``[l, n]`` is the fraction of *normal_prompts* for which neuron
            ``(l, n)`` exceeded *t_quiet*.
        """
        if not normal_prompts:
            return np.zeros(
                (self._n_layers, self._n_neurons), dtype=np.float32
            )
        acts = self._run_prompts(normal_prompts)
        margins = exceedance_margin(acts, self._stats, std_floor=self.std_floor)
        # fire_rates shape: (n_layers, n_neurons)
        fire_rates = np.mean(margins > self.t_quiet, axis=0).astype(np.float32)
        return fire_rates

    @property
    def quiet_threshold_rate(self) -> float:
        """Maximum allowed normal-fire-rate for a neuron to pass the quiet gate."""
        return 1.0 - self.quiet_pass_min
