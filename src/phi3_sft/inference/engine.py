"""Inference engines: vLLM (production) and Transformers (fallback)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from phi3_sft.config import InferenceConfig


class BaseInferenceEngine(ABC):
    """Common interface for inference engines."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response for a single prompt."""

    @abstractmethod
    def batch_generate(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts."""


class VLLMInferenceEngine(BaseInferenceEngine):
    """vLLM-based inference engine for high-throughput production use.

    Requires a *merged* model checkpoint (base + LoRA fused).
    Alternatively, pass ``enable_lora=True`` in ``InferenceConfig`` and set
    ``lora_path`` to hot-swap a LoRA adapter at runtime.
    """

    def __init__(self, config: InferenceConfig) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vllm is not installed. Install with: uv sync --group inference"
            ) from e

        self.config = config

        llm_kwargs: dict[str, Any] = {
            "model": config.model_path,
            "tensor_parallel_size": config.tensor_parallel_size,
            "trust_remote_code": True,
        }
        if config.enable_lora:
            llm_kwargs["enable_lora"] = True

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
        )

    def generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    def batch_generate(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text for o in outputs]


class TransformersInferenceEngine(BaseInferenceEngine):
    """HuggingFace Transformers inference engine — CPU/MPS/any-GPU fallback.

    Loads a merged model directory **or** a base model + LoRA adapter via PEFT.
    """

    def __init__(self, config: InferenceConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config

        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        model_path = config.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if config.enable_lora and config.lora_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, config.lora_path)

        if self.device != "cuda":
            model = model.to(self.device)

        self.model = model

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _run(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with __import__("torch").no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only the generated tokens (skip the prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate(self, prompt: str) -> str:
        return self._run(prompt)

    def batch_generate(self, prompts: list[str]) -> list[str]:
        return [self._run(p) for p in prompts]


def get_engine(config: InferenceConfig) -> BaseInferenceEngine:
    """Factory: return the appropriate engine based on ``config.engine``."""
    if config.engine == "vllm":
        return VLLMInferenceEngine(config)
    if config.engine == "transformers":
        return TransformersInferenceEngine(config)
    raise ValueError(f"Unknown engine '{config.engine}'. Choose 'vllm' or 'transformers'.")
