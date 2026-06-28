from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branch.data.dataset_specs import default_processed_dir, normalize_dataset_name
from branch.data.preprocessors import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BRANCH datasets.")
    parser.add_argument(
        "--dataset",
        default="maternal_health",
        help="Dataset to preprocess: gallstone, maternal_health, npha, load_diabetes, or all.",
    )
    parser.add_argument("--raw-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = (
        ["gallstone", "maternal_health", "npha", "load_diabetes"]
        if args.dataset == "all"
        else [normalize_dataset_name(args.dataset)]
    )
    for dataset in datasets:
        output_dir = args.output_dir or default_processed_dir(dataset)
        summary = preprocess_dataset(
            dataset=dataset,
            raw_path=args.raw_path if len(datasets) == 1 else None,
            output_dir=output_dir,
            test_size=args.test_size,
            seed=args.seed,
        )
        print(summary)


if __name__ == "__main__":
    main()
