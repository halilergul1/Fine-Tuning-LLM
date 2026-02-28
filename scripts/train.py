"""CLI: run SFT fine-tuning.

Usage::

    # Full training + adapter merge (requires Linux + CUDA GPU, ~8GB VRAM with 4-bit QLoRA)
    uv run --group train python scripts/train.py \\
        --model-id microsoft/Phi-3-mini-4k-instruct \\
        --epochs 3 \\
        --merge

    # Custom LoRA rank and batch size
    uv run --group train python scripts/train.py --lora-r 32 --batch-size 4 --output-dir outputs/run2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from phi3_sft.config import DataConfig, LoRAConfig, ModelConfig, TrainingConfig
from phi3_sft.training.trainer import SFTTrainerWrapper


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT fine-tuning on Phi-3-mini.")

    # Model
    parser.add_argument("--model-id", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--local-model-dir", default=None, help="Use a local model directory")
    parser.add_argument(
        "--quantization", choices=["4bit", "8bit", "none"], default="4bit"
    )
    parser.add_argument(
        "--no-unsloth", action="store_true", help="Disable Unsloth (use vanilla HF)"
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Training
    parser.add_argument("--output-dir", default="outputs/phi3-sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Max token length. Reduce to 512 on macOS/16GB RAM when fine-tuning large models.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Cap training at N steps (-1 = use --epochs). Useful for smoke tests.",
    )
    parser.add_argument(
        "--report-to", choices=["tensorboard", "wandb", "none"], default="tensorboard"
    )

    # Data
    parser.add_argument("--raw-path", default="data/mock_dataset.json")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument(
        "--format", dest="format_style", choices=["phi3", "alpaca"], default="phi3"
    )

    # Post-training
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA adapter into base model after training (needed for vLLM)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    model_config = ModelConfig(
        model_id=args.model_id,
        local_dir=args.local_model_dir,
        quantization=args.quantization,
        use_unsloth=not args.no_unsloth,
    )
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    train_config = TrainingConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        max_steps=args.max_steps,
        report_to=args.report_to,
    )
    data_config = DataConfig(
        raw_path=args.raw_path,
        processed_dir=args.processed_dir,
        format_style=args.format_style,
    )

    wrapper = SFTTrainerWrapper(model_config, lora_config, train_config, data_config)
    wrapper.setup()
    output_dir = wrapper.train()
    adapter_path = wrapper.save_adapter()
    print(f"Training complete. Adapter saved to: {adapter_path}")

    if args.merge:
        merged_path = wrapper.merge_and_export()
        print(f"Merged model saved to: {merged_path}")
        print("You can now run inference with:")
        print(f"  uv run --group inference python scripts/inference.py --model-path {merged_path}")


if __name__ == "__main__":
    main()
