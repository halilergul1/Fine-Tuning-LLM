"""Configuration dataclasses for phi3-sft."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    local_dir: str | None = None  # override to use a local path
    quantization: str = "4bit"    # "4bit" | "8bit" | "none"
    use_unsloth: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/phi3-sft"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"    # unsloth default
    bf16: bool = True
    max_steps: int = -1          # -1 = use num_train_epochs; set >0 to cap steps (useful for smoke tests)
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    seed: int = 42
    report_to: str = "tensorboard"


@dataclass
class DataConfig:
    raw_path: str = "data/mock_dataset.json"
    processed_dir: str = "data/processed"
    format_style: str = "phi3"   # "phi3" | "alpaca"
    train_ratio: float = 0.8
    val_ratio: float = 0.1       # remainder → test


@dataclass
class InferenceConfig:
    model_path: str = "outputs/phi3-sft/merged"
    engine: str = "vllm"         # "vllm" | "transformers"
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    tensor_parallel_size: int = 1  # vLLM: number of GPUs
    enable_lora: bool = False      # pass LoRA adapter dir instead of merged model
    lora_path: str | None = None   # path to adapter dir (used when enable_lora=True)
