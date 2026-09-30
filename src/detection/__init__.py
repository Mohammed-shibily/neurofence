"""NeuroFence detection package."""

from .baseline import (
    calibrate,
    calibrate_exceedance,
    collect_max_activations,
    compute_baseline_stats,
    exceedance_margin,
    format_report,
    load_baseline,
    save_baseline,
)

__all__ = [
    "collect_max_activations",
    "compute_baseline_stats",
    "save_baseline",
    "load_baseline",
    "calibrate",
    "exceedance_margin",
    "calibrate_exceedance",
    "format_report",
]
