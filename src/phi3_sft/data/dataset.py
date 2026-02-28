"""Dataset loading, splitting, and JSONL persistence."""

from __future__ import annotations

import json
import os
import random

from datasets import Dataset

from phi3_sft.config import DataConfig
from phi3_sft.data.formatter import AlpacaFormatter, BaseFormatter, Phi3ChatMLFormatter


def _get_formatter(format_style: str) -> BaseFormatter:
    if format_style == "phi3":
        return Phi3ChatMLFormatter()
    if format_style == "alpaca":
        return AlpacaFormatter()
    raise ValueError(f"Unknown format_style '{format_style}'. Choose 'phi3' or 'alpaca'.")


class DatasetLoader:
    """Load raw JSON/JSONL data, format, split, and save processed splits."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.formatter = _get_formatter(config.format_style)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_raw(self) -> list[dict]:
        """Load raw examples from a JSON or JSONL file."""
        path = self.config.raw_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw data file not found: {path}")

        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Raw data must be a JSON array of objects.")
        return data

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(
        self, data: list[dict], seed: int = 42
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Shuffle and split data into (train, val, test) lists."""
        rng = random.Random(seed)
        data = data.copy()
        rng.shuffle(data)

        n = len(data)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)

        train = data[:n_train]
        val = data[n_train : n_train + n_val]
        test = data[n_train + n_val :]
        return train, val, test

    # ------------------------------------------------------------------
    # Formatting + saving
    # ------------------------------------------------------------------

    def save_splits(
        self,
        train: list[dict],
        val: list[dict],
        test: list[dict],
    ) -> None:
        """Format examples and write JSONL files to ``processed_dir``."""
        os.makedirs(self.config.processed_dir, exist_ok=True)
        splits = {"train": train, "val": val, "test": test}
        for name, raw in splits.items():
            formatted = self.formatter.format_dataset(raw)
            out_path = os.path.join(self.config.processed_dir, f"{name}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for ex in formatted:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"Saved {len(formatted)} {name} examples → {out_path}")

    # ------------------------------------------------------------------
    # Loading processed splits
    # ------------------------------------------------------------------

    def load_splits(self) -> tuple[Dataset, Dataset, Dataset]:
        """Load previously saved JSONL splits as HuggingFace Datasets."""
        datasets = []
        for name in ("train", "val", "test"):
            path = os.path.join(self.config.processed_dir, f"{name}.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Processed split not found: {path}. "
                    "Run prepare_data.py first."
                )
            examples = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
            datasets.append(Dataset.from_list(examples))
        return tuple(datasets)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Convenience: prepare everything in one call
    # ------------------------------------------------------------------

    def prepare(self, seed: int = 42) -> tuple[Dataset, Dataset, Dataset]:
        """Load raw → split → format → save → return HF Datasets."""
        raw = self.load_raw()
        train_raw, val_raw, test_raw = self.split(raw, seed=seed)
        self.save_splits(train_raw, val_raw, test_raw)
        return self.load_splits()
