"""Command-line entry point for the offline predictive-maintenance demo."""

from __future__ import annotations

import argparse
import json

from src.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the beginner-friendly command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate and model synthetic IoT predictive-maintenance data."
    )
    parser.add_argument("--machines", type=int, default=6, help="Number of simulated machines.")
    parser.add_argument("--cycles", type=int, default=240, help="Readings per machine.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for CSV, JSON, model, and plot outputs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for failure alerts (0 to 1).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG generation when only tabular/model outputs are needed.",
    )
    return parser


def main() -> None:
    """Run the full demo and print its JSON-compatible summary."""
    args = build_parser().parse_args()
    summary = run_pipeline(
        machines=args.machines,
        cycles=args.cycles,
        seed=args.seed,
        output_dir=args.output_dir,
        threshold=args.threshold,
        skip_plots=args.skip_plots,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
