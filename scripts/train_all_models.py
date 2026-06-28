from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branch.data.dataset_specs import default_processed_dir, normalize_dataset_name
from branch.data.loaders import load_processed_split
from branch.models.train_baselines import train_and_evaluate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BRANCH baseline models.")
    parser.add_argument(
        "--dataset",
        default="maternal_health",
        help="Dataset to train: gallstone, maternal_health, npha, load_diabetes, or all.",
    )
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = (
        ["gallstone", "maternal_health", "npha", "load_diabetes"]
        if args.dataset == "all"
        else [normalize_dataset_name(args.dataset)]
    )
    for dataset in datasets:
        processed_dir = args.processed_dir or default_processed_dir(dataset)
        train_df, test_df = load_processed_split(dataset, processed_dir)
        metrics = train_and_evaluate_dataset(dataset, train_df, test_df, seed=args.seed)
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
