#!/usr/bin/env python3
"""Download or materialize benchmark datasets.

By default this only warms the standard Hugging Face cache. If ``--output_dir``
is provided, each dataset is also saved with ``save_to_disk`` for offline use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    "english": "nyrahealth/disfluency_speech_english",
    "german": "nyrahealth/disfluency_speech_german",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS) + ["all"], default=["all"])
    parser.add_argument("--output_dir", default="", help="Optional directory for save_to_disk copies.")
    args = parser.parse_args()

    selected = list(DATASETS) if "all" in args.datasets else args.datasets
    output_dir = Path(args.output_dir) if args.output_dir else None

    for key in selected:
        name = DATASETS[key]
        print(f"Loading {name}...")
        ds = load_dataset(name)
        print(ds)
        if output_dir:
            target = output_dir / key
            target.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(target))
            print(f"Saved {name} to {target}")


if __name__ == "__main__":
    main()
