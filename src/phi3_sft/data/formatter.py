"""Prompt formatters for SFT fine-tuning.

Each formatter converts a raw (instruction, input, response) triple into a
single ``text`` string that SFTTrainer consumes via ``dataset_text_field="text"``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseFormatter(ABC):
    """Common interface for prompt formatters."""

    @abstractmethod
    def format_example(self, instruction: str, response: str, input_ctx: str = "") -> str:
        """Format one example into a single prompt string."""

    def format_dataset(self, examples: list[dict]) -> list[dict]:
        """Format a list of raw dicts into ``{"text": ...}`` dicts.

        Each raw dict must have ``"instruction"`` and ``"response"`` keys;
        ``"input"`` is optional.
        """
        formatted = []
        for ex in examples:
            text = self.format_example(
                instruction=ex["instruction"],
                response=ex["response"],
                input_ctx=ex.get("input", ""),
            )
            formatted.append({"text": text})
        return formatted


class Phi3ChatMLFormatter(BaseFormatter):
    """Phi-3 native ChatML template.

    Format::

        <|user|>
        {instruction}
        {input_ctx}
        <|end|>
        <|assistant|>
        {response}<|end|>

    When ``input_ctx`` is empty the blank line is omitted.
    """

    def format_example(self, instruction: str, response: str, input_ctx: str = "") -> str:
        prompt_parts = [instruction]
        if input_ctx.strip():
            prompt_parts.append(input_ctx.strip())
        prompt = "\n".join(prompt_parts)
        return f"<|user|>\n{prompt}\n<|end|>\n<|assistant|>\n{response}<|end|>"


class AlpacaFormatter(BaseFormatter):
    """Classic Alpaca instruction-following template.

    Format::

        Below is an instruction that describes a task...

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {response}

    ``### Input:`` section is omitted when ``input_ctx`` is empty.
    """

    _PREFIX = (
        "Below is an instruction that describes a task, paired with an input that provides "
        "further context. Write a response that appropriately completes the request."
    )
    _PREFIX_NO_INPUT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    def format_example(self, instruction: str, response: str, input_ctx: str = "") -> str:
        if input_ctx.strip():
            return (
                f"{self._PREFIX}\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_ctx.strip()}\n\n"
                f"### Response:\n{response}"
            )
        return (
            f"{self._PREFIX_NO_INPUT}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )
