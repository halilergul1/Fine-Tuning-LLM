"""Unit tests for AlpacaFormatter and Phi3ChatMLFormatter."""

import sys
from pathlib import Path

# Ensure the src/ tree is importable without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from phi3_sft.data.formatter import AlpacaFormatter, Phi3ChatMLFormatter


# ---------------------------------------------------------------------------
# Phi3ChatMLFormatter
# ---------------------------------------------------------------------------

class TestPhi3ChatMLFormatter:
    def setup_method(self):
        self.fmt = Phi3ChatMLFormatter()

    def test_format_without_input(self):
        text = self.fmt.format_example(
            instruction="Write a function to add two numbers.",
            response="def add(a, b):\n    return a + b",
        )
        assert text.startswith("<|user|>")
        assert "Write a function to add two numbers." in text
        assert "<|end|>" in text
        assert "<|assistant|>" in text
        assert "def add(a, b):" in text
        assert text.endswith("<|end|>")

    def test_format_with_input(self):
        text = self.fmt.format_example(
            instruction="Write a sorting function.",
            response="def sort(lst):\n    return sorted(lst)",
            input_ctx="The function should sort in ascending order.",
        )
        assert "The function should sort in ascending order." in text
        assert "<|user|>" in text
        assert "<|assistant|>" in text

    def test_format_empty_input_ctx_omitted(self):
        text_no_input = self.fmt.format_example("Do X", "Result X", input_ctx="")
        text_whitespace = self.fmt.format_example("Do X", "Result X", input_ctx="   ")
        assert text_no_input == text_whitespace

    def test_format_dataset(self):
        examples = [
            {"instruction": "Instr 1", "input": "", "response": "Resp 1"},
            {"instruction": "Instr 2", "input": "ctx", "response": "Resp 2"},
        ]
        formatted = self.fmt.format_dataset(examples)
        assert len(formatted) == 2
        for item in formatted:
            assert "text" in item
            assert isinstance(item["text"], str)
            assert "<|user|>" in item["text"]

    def test_template_structure(self):
        text = self.fmt.format_example("Say hello", "Hello!")
        # Verify ordering: user block before assistant block
        user_idx = text.index("<|user|>")
        asst_idx = text.index("<|assistant|>")
        assert user_idx < asst_idx

    def test_response_in_assistant_block(self):
        response = "def foo(): pass"
        text = self.fmt.format_example("Write foo", response)
        asst_idx = text.index("<|assistant|>")
        assert response in text[asst_idx:]


# ---------------------------------------------------------------------------
# AlpacaFormatter
# ---------------------------------------------------------------------------

class TestAlpacaFormatter:
    def setup_method(self):
        self.fmt = AlpacaFormatter()

    def test_format_without_input(self):
        text = self.fmt.format_example(
            instruction="Describe recursion.",
            response="Recursion is a function calling itself.",
        )
        assert "### Instruction:" in text
        assert "Describe recursion." in text
        assert "### Response:" in text
        assert "Recursion is a function calling itself." in text
        assert "### Input:" not in text

    def test_format_with_input(self):
        text = self.fmt.format_example(
            instruction="Sort the following list.",
            response="[1, 2, 3]",
            input_ctx="[3, 1, 2]",
        )
        assert "### Input:" in text
        assert "[3, 1, 2]" in text
        assert "[1, 2, 3]" in text

    def test_prefix_differs_with_and_without_input(self):
        text_with = self.fmt.format_example("I", "R", input_ctx="ctx")
        text_without = self.fmt.format_example("I", "R", input_ctx="")
        # Different preambles
        assert text_with.split("\n")[0] != text_without.split("\n")[0]

    def test_format_dataset(self):
        examples = [
            {"instruction": "Do A", "input": "", "response": "A done"},
            {"instruction": "Do B", "input": "with context", "response": "B done"},
        ]
        formatted = self.fmt.format_dataset(examples)
        assert len(formatted) == 2
        for item in formatted:
            assert "text" in item
            assert "### Instruction:" in item["text"]

    def test_ordering(self):
        text = self.fmt.format_example("I", "R", input_ctx="ctx")
        instr_idx = text.index("### Instruction:")
        input_idx = text.index("### Input:")
        resp_idx = text.index("### Response:")
        assert instr_idx < input_idx < resp_idx
