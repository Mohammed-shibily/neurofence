"""NeuroFence CLI — scan a model directory for backdoor neurons.

Usage
-----
    python run_scan.py <model_dir> [options]

Options
-------
    --baseline PATH         Path to baseline_clean.npz
                            (default: data/baseline_clean.npz)
    --words WORD [WORD ...]
                            Override the default candidate word list.
                            Default: TRIGGER_WORDS + CONTROL_WORDS from
                            src/fuzzer/generator.py
    --t-probe FLOAT         Exceedance-margin threshold for consistency gate
                            (default: 20.0)
    --t-quiet FLOAT         Exceedance-margin threshold for quiet gate
                            (default: 8.0)
    --normal-n INT          Number of normal prompts for quiet check
                            (default: 200)
    --seed INT              Master random seed (default: 42)
    --no-progress           Suppress per-word progress output

Exit codes
----------
    0   CLEAN
    1   BACKDOOR DETECTED or error
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path when run as a top-level script.
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.detection.analyzer import (
    DEFAULT_NORMAL_SAMPLE_N,
    scan_model,
)
from src.detection.probe import DEFAULT_T_PROBE, DEFAULT_T_QUIET


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 70
_DASH = "-" * 70


def _fmt_report(result: dict, model_dir: str) -> str:
    """Format the scan result dictionary into a human-readable string.

    Args:
        result: Dictionary returned by :func:`~src.detection.analyzer.scan_model`.
        model_dir: Model directory path string (for display).

    Returns:
        Multi-line formatted report string.
    """
    lines = [
        _SEP,
        "NeuroFence Backdoor Scan",
        _SEP,
        f"Model        : {model_dir}",
        f"Hash         : {result['model_hash'][:16]}...{result['model_hash'][-8:]}",
        f"Prompts tested : {result['prompts_tested']}",
        _DASH,
    ]

    flagged = result["flagged_neurons"]
    if flagged:
        lines.append(f"FLAGGED NEURONS ({len(flagged)}):")
        for entry in flagged:
            lines.append(
                f"  \u2022 Layer {entry['layer']}, Neuron {entry['neuron']}"
                f"  |  word='{entry['word']}'"
                f"  consistency={entry['consistency']:.2f}"
                f"  median_margin={entry['median_margin']:.2f}"
                f"  normal_fire_rate={entry['normal_fire_rate']:.4f}"
            )
    else:
        lines.append("FLAGGED NEURONS: none")

    lines += [
        _DASH,
        f"Safety Score : {result['safety_score']}/100",
        f"Verdict      : {result['verdict']}",
        _DASH,
        "Limitations:",
    ]
    for i, lim in enumerate(result["limitations"], 1):
        # Word-wrap at 66 chars
        words = lim.split()
        current_line = f"  [{i}] "
        indent = " " * 6
        for word in words:
            if len(current_line) + len(word) + 1 > 70:
                lines.append(current_line)
                current_line = indent + word
            else:
                current_line = (current_line + " " + word).lstrip()
                if current_line == word:
                    current_line = f"  [{i}] " + word
        if current_line.strip():
            lines.append(current_line)

    lines.append(_SEP)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Parse arguments, run the scan, print the report, and return an exit code.

    Returns:
        0 if the verdict is CLEAN, 1 if BACKDOOR DETECTED or on error.
    """
    parser = argparse.ArgumentParser(
        prog="run_scan.py",
        description=(
            "NeuroFence: scan a local GPT-2 model directory for backdoor neurons."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model_dir",
        help="Path to the model directory (must contain config.json + .safetensors).",
    )
    parser.add_argument(
        "--baseline",
        default=str(Path(__file__).resolve().parent / "data" / "baseline_clean.npz"),
        metavar="PATH",
        help="Path to baseline_clean.npz (default: data/baseline_clean.npz).",
    )
    parser.add_argument(
        "--words",
        nargs="+",
        default=None,
        metavar="WORD",
        help="Override the default candidate word list.",
    )
    parser.add_argument(
        "--t-probe",
        type=float,
        default=DEFAULT_T_PROBE,
        metavar="FLOAT",
        help=f"Consistency-gate threshold (default: {DEFAULT_T_PROBE}).",
    )
    parser.add_argument(
        "--t-quiet",
        type=float,
        default=DEFAULT_T_QUIET,
        metavar="FLOAT",
        help=f"Quiet-gate threshold (default: {DEFAULT_T_QUIET}).",
    )
    parser.add_argument(
        "--normal-n",
        type=int,
        default=DEFAULT_NORMAL_SAMPLE_N,
        metavar="INT",
        help=f"Number of normal prompts for quiet check (default: {DEFAULT_NORMAL_SAMPLE_N}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Master random seed (default: 42).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-word progress output.",
    )

    args = parser.parse_args(argv)

    # Reconfigure stdout to UTF-8 on Windows if possible
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    try:
        result = scan_model(
            model_dir=args.model_dir,
            baseline_path=args.baseline,
            words=args.words,
            t_probe=args.t_probe,
            t_quiet=args.t_quiet,
            normal_sample_n=args.normal_n,
            seed=args.seed,
            progress=not args.no_progress,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERROR] Scan failed: {exc!r}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print(_fmt_report(result, args.model_dir))
    return 0 if result["verdict"] == "CLEAN" else 1


if __name__ == "__main__":
    sys.exit(main())
