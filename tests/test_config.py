"""Unit tests for configuration dataclasses."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from phi3_sft.config import (
    DataConfig,
    InferenceConfig,
    LoRAConfig,
    ModelConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.model_id == "microsoft/Phi-3-mini-4k-instruct"
        assert cfg.local_dir is None
        assert cfg.quantization == "4bit"
        assert cfg.use_unsloth is True

    def test_custom_values(self):
        cfg = ModelConfig(
            model_id="meta-llama/Llama-3-8B",
            local_dir="/tmp/model",
            quantization="8bit",
            use_unsloth=False,
        )
        assert cfg.model_id == "meta-llama/Llama-3-8B"
        assert cfg.local_dir == "/tmp/model"
        assert cfg.quantization == "8bit"
        assert cfg.use_unsloth is False

    def test_dataclass_is_mutable(self):
        cfg = ModelConfig()
        cfg.model_id = "new-model"
        assert cfg.model_id == "new-model"


# ---------------------------------------------------------------------------
# LoRAConfig
# ---------------------------------------------------------------------------

class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert "q_proj" in cfg.target_modules
        assert "v_proj" in cfg.target_modules
        assert cfg.bias == "none"
        assert cfg.task_type == "CAUSAL_LM"

    def test_target_modules_are_independent(self):
        # Each instance should have its own list (no shared mutable default)
        cfg1 = LoRAConfig()
        cfg2 = LoRAConfig()
        cfg1.target_modules.append("extra_proj")
        assert "extra_proj" not in cfg2.target_modules

    def test_custom_r(self):
        cfg = LoRAConfig(r=64, lora_alpha=128)
        assert cfg.r == 64
        assert cfg.lora_alpha == 128

    def test_all_default_target_modules_present(self):
        expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        cfg = LoRAConfig()
        assert expected == set(cfg.target_modules)


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.output_dir == "outputs/phi3-sft"
        assert cfg.per_device_train_batch_size == 2
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.learning_rate == pytest.approx(2e-4)
        assert cfg.num_train_epochs == 3
        assert cfg.max_seq_length == 2048
        assert cfg.warmup_ratio == pytest.approx(0.05)
        assert cfg.lr_scheduler_type == "cosine"
        assert cfg.optim == "adamw_8bit"
        assert cfg.bf16 is True
        assert cfg.seed == 42
        assert cfg.report_to == "tensorboard"

    def test_custom_output_dir(self):
        cfg = TrainingConfig(output_dir="/tmp/my-run")
        assert cfg.output_dir == "/tmp/my-run"

    def test_effective_batch_size(self):
        cfg = TrainingConfig(per_device_train_batch_size=4, gradient_accumulation_steps=8)
        effective = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
        assert effective == 32


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.raw_path == "data/mock_dataset.json"
        assert cfg.processed_dir == "data/processed"
        assert cfg.format_style == "phi3"
        assert cfg.train_ratio == pytest.approx(0.8)
        assert cfg.val_ratio == pytest.approx(0.1)

    def test_test_ratio_implied(self):
        cfg = DataConfig(train_ratio=0.7, val_ratio=0.2)
        test_ratio = 1.0 - cfg.train_ratio - cfg.val_ratio
        assert test_ratio == pytest.approx(0.1)

    def test_format_style_alpaca(self):
        cfg = DataConfig(format_style="alpaca")
        assert cfg.format_style == "alpaca"


# ---------------------------------------------------------------------------
# InferenceConfig
# ---------------------------------------------------------------------------

class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.model_path == "outputs/phi3-sft/merged"
        assert cfg.engine == "vllm"
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == pytest.approx(0.1)
        assert cfg.top_p == pytest.approx(0.9)
        assert cfg.tensor_parallel_size == 1
        assert cfg.enable_lora is False
        assert cfg.lora_path is None

    def test_transformers_engine(self):
        cfg = InferenceConfig(engine="transformers")
        assert cfg.engine == "transformers"

    def test_lora_fields(self):
        cfg = InferenceConfig(enable_lora=True, lora_path="/tmp/adapter")
        assert cfg.enable_lora is True
        assert cfg.lora_path == "/tmp/adapter"

    def test_multi_gpu(self):
        cfg = InferenceConfig(tensor_parallel_size=4)
        assert cfg.tensor_parallel_size == 4
