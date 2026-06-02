#!/usr/bin/env python3
"""Reproduce cached benchmark results with one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_step(command: list[str], label: str) -> None:
    print(f"\n==> {label}")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default="results/cached",
        help="Directory where reports and leaderboard are written.",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip prediction validation before evaluation.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    leaderboard_path = output_dir / "leaderboard.md"

    if not args.skip_validation:
        run_step(
            [sys.executable, "scripts/validate_predictions.py"],
            "Validating cached predictions",
        )

    run_step(
        [sys.executable, "scripts/convert_outputs.py", "--skip_import"],
        "Regenerating normalized predictions",
    )
    run_step(
        [
            sys.executable,
            "scripts/evaluate_all_cached.py",
            "--output_dir",
            str(output_dir),
        ],
        "Evaluating cached predictions",
    )
    run_step(
        [
            sys.executable,
            "scripts/print_leaderboard.py",
            "--output",
            str(leaderboard_path),
        ],
        "Writing leaderboard",
    )

    print(f"\nDone. Leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()
