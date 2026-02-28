"""CLI: run inference on a fine-tuned model.

Usage::

    # Single prompt with vLLM
    uv run --group inference python scripts/inference.py \\
        --model-path outputs/phi3-sft/merged \\
        --engine vllm \\
        --prompt "Write a Python function to sort a list"

    # Transformers fallback (macOS / CPU)
    uv run python scripts/inference.py \\
        --engine transformers \\
        --prompt "Write a Python function to reverse a string"

    # Batch from a file (one prompt per line)
    uv run python scripts/inference.py \\
        --engine transformers \\
        --batch-file prompts.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phi3_sft.config import InferenceConfig
from phi3_sft.inference.engine import get_engine


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a fine-tuned Phi-3 model.")

    parser.add_argument(
        "--model-path",
        default="outputs/phi3-sft/merged",
        help="Path to merged model dir (default: outputs/phi3-sft/merged)",
    )
    parser.add_argument(
        "--engine",
        choices=["vllm", "transformers"],
        default="transformers",
        help="Inference backend (default: transformers)",
    )
    parser.add_argument("--prompt", default=None, help="Single prompt string")
    parser.add_argument(
        "--batch-file",
        default=None,
        help="Path to a text file with one prompt per line",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism",
    )
    parser.add_argument(
        "--enable-lora",
        action="store_true",
        help="Load LoRA adapter at runtime (vLLM only)",
    )
    parser.add_argument("--lora-path", default=None, help="Path to LoRA adapter directory")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.prompt is None and args.batch_file is None:
        print("Error: provide --prompt or --batch-file", file=sys.stderr)
        sys.exit(1)

    config = InferenceConfig(
        model_path=args.model_path,
        engine=args.engine,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_lora=args.enable_lora,
        lora_path=args.lora_path,
    )

    engine = get_engine(config)

    if args.batch_file:
        prompts = Path(args.batch_file).read_text().splitlines()
        prompts = [p.strip() for p in prompts if p.strip()]
        print(f"Running batch inference on {len(prompts)} prompts …\n")
        responses = engine.batch_generate(prompts)
        for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
            print(f"--- [{i}] Prompt ---\n{prompt}")
            print(f"--- [{i}] Response ---\n{response}\n")
    else:
        print(f"Prompt: {args.prompt}\n")
        response = engine.generate(args.prompt)
        print(f"Response:\n{response}")


if __name__ == "__main__":
    main()
