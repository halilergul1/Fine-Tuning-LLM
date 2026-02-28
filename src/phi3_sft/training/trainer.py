"""SFT fine-tuning wrapper using TRL's SFTTrainer."""

from __future__ import annotations

import os

from phi3_sft.config import DataConfig, LoRAConfig, ModelConfig, TrainingConfig
from phi3_sft.data.dataset import DatasetLoader
from phi3_sft.model.loader import ModelLoader


class SFTTrainerWrapper:
    """Orchestrate the full SFT pipeline: data → model → train → save."""

    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        train_config: TrainingConfig,
        data_config: DataConfig,
    ) -> None:
        self.model_config = model_config
        self.lora_config = lora_config
        self.train_config = train_config
        self.data_config = data_config

        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Load model, tokenizer, and datasets."""
        loader = ModelLoader(self.model_config)
        self.model, self.tokenizer = loader.load(lora_config=self.lora_config)

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load (or prepare) processed splits
        data_loader = DatasetLoader(self.data_config)
        processed_train = os.path.join(self.data_config.processed_dir, "train.jsonl")
        if os.path.exists(processed_train):
            self.train_dataset, self.eval_dataset, _ = data_loader.load_splits()
        else:
            self.train_dataset, self.eval_dataset, _ = data_loader.prepare()

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self) -> str:
        """Run SFT training and return the output directory path."""
        if self.model is None:
            raise RuntimeError("Call setup() before train().")

        try:
            from trl import SFTConfig, SFTTrainer
        except ImportError as e:
            raise ImportError(
                "trl is not installed. Install with: uv sync --group train"
            ) from e

        import torch

        use_bf16 = self.train_config.bf16 and torch.cuda.is_available()
        use_fp16 = (
            not use_bf16
            and torch.cuda.is_available()
            and not torch.backends.mps.is_available()
        )
        # adamw_8bit requires bitsandbytes + CUDA; fall back to standard adamw on MPS/CPU
        optim = self.train_config.optim if torch.cuda.is_available() else "adamw_torch"

        sft_config = SFTConfig(
            output_dir=self.train_config.output_dir,
            per_device_train_batch_size=self.train_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            learning_rate=self.train_config.learning_rate,
            num_train_epochs=self.train_config.num_train_epochs,
            max_steps=self.train_config.max_steps,  # -1 means use num_train_epochs
            max_length=self.train_config.max_seq_length,  # renamed in TRL 0.12+
            warmup_ratio=self.train_config.warmup_ratio,
            lr_scheduler_type=self.train_config.lr_scheduler_type,
            optim=optim,
            bf16=use_bf16,
            fp16=use_fp16,
            logging_steps=self.train_config.logging_steps,
            eval_steps=self.train_config.eval_steps,
            save_steps=self.train_config.save_steps,
            seed=self.train_config.seed,
            report_to=self.train_config.report_to,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,  # renamed from tokenizer in TRL 0.13+
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=sft_config,
        )

        print("Starting SFT training …")
        trainer.train()
        return self.train_config.output_dir

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_adapter(self) -> str:
        """Save the LoRA adapter to ``{output_dir}/adapter``."""
        if self.model is None:
            raise RuntimeError("Call setup() and train() before save_adapter().")

        adapter_path = os.path.join(self.train_config.output_dir, "adapter")
        os.makedirs(adapter_path, exist_ok=True)
        self.model.save_pretrained(adapter_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(adapter_path)
        print(f"LoRA adapter saved → {adapter_path}")
        return adapter_path

    def merge_and_export(self) -> str:
        """Merge LoRA adapter into the base model and save for vLLM."""
        adapter_path = os.path.join(self.train_config.output_dir, "adapter")
        merged_path = os.path.join(self.train_config.output_dir, "merged")

        model_loader = ModelLoader(self.model_config)
        model_loader.merge_and_save(adapter_path=adapter_path, output_path=merged_path)
        return merged_path
