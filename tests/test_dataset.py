"""Unit tests for DatasetLoader."""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from phi3_sft.config import DataConfig
from phi3_sft.data.dataset import DatasetLoader

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATA = [
    {"instruction": f"Instruction {i}", "input": "", "response": f"Response {i}"}
    for i in range(20)
]


@pytest.fixture
def raw_json_file(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(SAMPLE_DATA), encoding="utf-8")
    return str(path)


@pytest.fixture
def raw_jsonl_file(tmp_path):
    path = tmp_path / "data.jsonl"
    lines = [json.dumps(ex) for ex in SAMPLE_DATA]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def phi3_config(raw_json_file, tmp_path):
    return DataConfig(
        raw_path=raw_json_file,
        processed_dir=str(tmp_path / "processed"),
        format_style="phi3",
        train_ratio=0.7,
        val_ratio=0.15,
    )


@pytest.fixture
def alpaca_config(raw_json_file, tmp_path):
    return DataConfig(
        raw_path=raw_json_file,
        processed_dir=str(tmp_path / "processed"),
        format_style="alpaca",
        train_ratio=0.8,
        val_ratio=0.1,
    )


# ---------------------------------------------------------------------------
# load_raw
# ---------------------------------------------------------------------------

class TestLoadRaw:
    def test_load_json(self, phi3_config):
        loader = DatasetLoader(phi3_config)
        data = loader.load_raw()
        assert len(data) == 20
        assert data[0]["instruction"] == "Instruction 0"

    def test_load_jsonl(self, raw_jsonl_file, tmp_path):
        config = DataConfig(
            raw_path=raw_jsonl_file,
            processed_dir=str(tmp_path / "p"),
        )
        loader = DatasetLoader(config)
        data = loader.load_raw()
        assert len(data) == 20

    def test_missing_file_raises(self, tmp_path):
        config = DataConfig(raw_path=str(tmp_path / "missing.json"))
        loader = DatasetLoader(config)
        with pytest.raises(FileNotFoundError):
            loader.load_raw()


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

class TestSplit:
    def test_split_sizes(self, phi3_config):
        loader = DatasetLoader(phi3_config)
        train, val, test = loader.split(SAMPLE_DATA)
        assert len(train) + len(val) + len(test) == len(SAMPLE_DATA)
        assert len(train) == int(20 * 0.7)
        assert len(val) == int(20 * 0.15)

    def test_split_is_deterministic(self, phi3_config):
        loader = DatasetLoader(phi3_config)
        t1, v1, te1 = loader.split(SAMPLE_DATA, seed=0)
        t2, v2, te2 = loader.split(SAMPLE_DATA, seed=0)
        assert t1 == t2 and v1 == v2 and te1 == te2

    def test_different_seeds_differ(self, phi3_config):
        loader = DatasetLoader(phi3_config)
        t1, _, _ = loader.split(SAMPLE_DATA, seed=1)
        t2, _, _ = loader.split(SAMPLE_DATA, seed=2)
        assert t1 != t2

    def test_no_overlap(self, phi3_config):
        loader = DatasetLoader(phi3_config)
        train, val, test = loader.split(SAMPLE_DATA)
        combined = train + val + test
        # All original elements present
        assert sorted(combined, key=lambda x: x["instruction"]) == sorted(
            SAMPLE_DATA, key=lambda x: x["instruction"]
        )


# ---------------------------------------------------------------------------
# save_splits / load_splits
# ---------------------------------------------------------------------------

class TestSaveSplits:
    def test_files_created(self, phi3_config):
        loader = DatasetLoader(phi3_config)
        train, val, test = loader.split(SAMPLE_DATA)
        loader.save_splits(train, val, test)

        for name in ("train", "val", "test"):
            path = os.path.join(phi3_config.processed_dir, f"{name}.jsonl")
            assert os.path.exists(path)

    def test_jsonl_content_has_text_field(self, phi3_config):
        loader = DatasetLoader(phi3_config)
        train, val, test = loader.split(SAMPLE_DATA)
        loader.save_splits(train, val, test)

        train_path = os.path.join(phi3_config.processed_dir, "train.jsonl")
        with open(train_path) as f:
            first = json.loads(f.readline())
        assert "text" in first
        assert isinstance(first["text"], str)

    def test_phi3_format_in_output(self, phi3_config):
        loader = DatasetLoader(phi3_config)
        train, val, test = loader.split(SAMPLE_DATA)
        loader.save_splits(train, val, test)

        train_path = os.path.join(phi3_config.processed_dir, "train.jsonl")
        with open(train_path) as f:
            first = json.loads(f.readline())
        assert "<|user|>" in first["text"]
        assert "<|assistant|>" in first["text"]

    def test_alpaca_format_in_output(self, alpaca_config):
        loader = DatasetLoader(alpaca_config)
        train, val, test = loader.split(SAMPLE_DATA)
        loader.save_splits(train, val, test)

        train_path = os.path.join(alpaca_config.processed_dir, "train.jsonl")
        with open(train_path) as f:
            first = json.loads(f.readline())
        assert "### Instruction:" in first["text"]


class TestLoadSplits:
    def test_returns_datasets(self, phi3_config):
        from datasets import Dataset

        loader = DatasetLoader(phi3_config)
        train, val, test = loader.split(SAMPLE_DATA)
        loader.save_splits(train, val, test)

        ds_train, ds_val, ds_test = loader.load_splits()
        assert isinstance(ds_train, Dataset)
        assert isinstance(ds_val, Dataset)
        assert isinstance(ds_test, Dataset)

    def test_dataset_has_text_column(self, phi3_config):
        loader = DatasetLoader(phi3_config)
        train, val, test = loader.split(SAMPLE_DATA)
        loader.save_splits(train, val, test)

        ds_train, _, _ = loader.load_splits()
        assert "text" in ds_train.column_names

    def test_missing_split_raises(self, phi3_config):
        loader = DatasetLoader(phi3_config)
        with pytest.raises(FileNotFoundError):
            loader.load_splits()


# ---------------------------------------------------------------------------
# prepare (end-to-end)
# ---------------------------------------------------------------------------

class TestPrepare:
    def test_prepare_end_to_end(self, phi3_config):
        from datasets import Dataset

        loader = DatasetLoader(phi3_config)
        train, val, test = loader.prepare()
        assert isinstance(train, Dataset)
        assert len(train) > 0
