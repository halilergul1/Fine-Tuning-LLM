"""CLI: format raw data → processed train/val/test splits.

Usage::

    uv run python scripts/prepare_data.py --raw-path data/mock_dataset.json --format phi3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phi3_sft.config import DataConfig
from phi3_sft.data.dataset import DatasetLoader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format and split raw data into train/val/test JSONL files."
    )
    parser.add_argument(
        "--raw-path",
        default="data/mock_dataset.json",
        help="Path to raw JSON or JSONL file (default: data/mock_dataset.json)",
    )
    parser.add_argument(
        "--format",
        dest="format_style",
        choices=["phi3", "alpaca"],
        default="phi3",
        help="Prompt template to use (default: phi3)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for processed JSONL splits (default: data/processed)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    config = DataConfig(
        raw_path=args.raw_path,
        processed_dir=args.output_dir,
        format_style=args.format_style,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    loader = DatasetLoader(config)
    raw = loader.load_raw()
    print(f"Loaded {len(raw)} raw examples from '{args.raw_path}'")

    train_raw, val_raw, test_raw = loader.split(raw, seed=args.seed)
    print(f"Split → train: {len(train_raw)}, val: {len(val_raw)}, test: {len(test_raw)}")

    loader.save_splits(train_raw, val_raw, test_raw)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
