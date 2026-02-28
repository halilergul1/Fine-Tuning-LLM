"""CLI: download a model from HuggingFace Hub.

Usage::

    uv run python scripts/download_model.py \\
        --model-id microsoft/Phi-3-mini-4k-instruct \\
        --local-dir models/phi3-mini

Requires HF_TOKEN in environment or .env file for gated models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from phi3_sft.config import ModelConfig
from phi3_sft.model.loader import ModelLoader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a model from HuggingFace Hub."
    )
    parser.add_argument(
        "--model-id",
        default="microsoft/Phi-3-mini-4k-instruct",
        help="HuggingFace model ID (default: microsoft/Phi-3-mini-4k-instruct)",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Local directory to save the model (default: use HF cache)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    config = ModelConfig(
        model_id=args.model_id,
        local_dir=args.local_dir,
    )
    loader = ModelLoader(config)
    path = loader.download()
    print(f"Model available at: {path}")


if __name__ == "__main__":
    main()
