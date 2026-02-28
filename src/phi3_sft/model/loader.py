"""Model downloading, loading, and LoRA-merging."""

from __future__ import annotations

import os
from typing import Any

from phi3_sft.config import LoRAConfig, ModelConfig


class ModelLoader:
    """Handle model download, loading (with optional LoRA), and adapter merging."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self) -> str:
        """Download model from HF Hub to ``local_dir`` (if set) and return the path/id to use."""
        if self.config.local_dir is None:
            return self.config.model_id

        from huggingface_hub import snapshot_download

        print(f"Downloading '{self.config.model_id}' → '{self.config.local_dir}' …")
        snapshot_download(
            repo_id=self.config.model_id,
            local_dir=self.config.local_dir,
            ignore_patterns=["*.msgpack", "flax_model*"],
        )
        return self.config.local_dir

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self, lora_config: LoRAConfig | None = None
    ) -> tuple[Any, Any]:
        """Load model + tokenizer, optionally wrapping with LoRA.

        Returns:
            (model, tokenizer)
        """
        model_path = self.config.local_dir if self.config.local_dir else self.config.model_id

        if self.config.use_unsloth:
            return self._load_unsloth(model_path, lora_config)
        return self._load_hf(model_path, lora_config)

    def _load_unsloth(self, model_path: str, lora_config: LoRAConfig | None) -> tuple[Any, Any]:
        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ImportError(
                "unsloth is not installed. Install with: uv sync --group train"
            ) from e

        load_in_4bit = self.config.quantization == "4bit"
        load_in_8bit = self.config.quantization == "8bit"

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )

        if lora_config is not None:
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                target_modules=lora_config.target_modules,
                bias=lora_config.bias,
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

        return model, tokenizer

    def _load_hf(self, model_path: str, lora_config: LoRAConfig | None) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        has_cuda = torch.cuda.is_available()
        has_mps = torch.backends.mps.is_available()

        quant_config = None
        if self.config.quantization == "4bit" and has_cuda:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.quantization == "8bit" and has_cuda:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.config.quantization != "none" and not has_cuda:
            print(
                f"Warning: quantization='{self.config.quantization}' requires CUDA. "
                "Loading in full precision instead."
            )

        # bfloat16 is only well-supported on CUDA; use float16 on MPS, float32 on CPU
        if quant_config is not None:
            dtype = None
        elif has_cuda:
            dtype = torch.bfloat16
        elif has_mps:
            dtype = torch.float16
        else:
            dtype = torch.float32

        device_map = "auto" if has_cuda else None

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=True,
        )

        if not has_cuda:
            device = "mps" if has_mps else "cpu"
            model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if lora_config is not None:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if self.config.quantization in ("4bit", "8bit"):
                model = prepare_model_for_kbit_training(model)

            peft_cfg = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                target_modules=lora_config.target_modules,
                bias=lora_config.bias,
                task_type=lora_config.task_type,
            )
            model = get_peft_model(model, peft_cfg)
            model.print_trainable_parameters()

        return model, tokenizer

    # ------------------------------------------------------------------
    # Merge & save
    # ------------------------------------------------------------------

    def merge_and_save(self, adapter_path: str, output_path: str) -> None:
        """Load base model + LoRA adapter, merge weights, save full model.

        The resulting directory is a standard HF checkpoint that vLLM can load.
        """
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.config.local_dir if self.config.local_dir else self.config.model_id
        print(f"Loading base model from '{model_path}' …")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f"Loading LoRA adapter from '{adapter_path}' …")
        model = PeftModel.from_pretrained(model, adapter_path)

        print("Merging adapter into base model …")
        model = model.merge_and_unload()

        os.makedirs(output_path, exist_ok=True)
        print(f"Saving merged model → '{output_path}' …")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print("Done.")
