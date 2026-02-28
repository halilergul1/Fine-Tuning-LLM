# Using This Project With Your Own Data

Four steps: format your data, train, get inference. That's it.

---

## 1. Prepare your dataset

Create a `.json` file — a list of objects with three keys:

```json
[
  {
    "instruction": "Classify the sentiment of this review.",
    "input": "The product broke after one day.",
    "response": "negative"
  },
  {
    "instruction": "Summarise the following paragraph in one sentence.",
    "input": "",
    "response": "..."
  }
]
```

- `instruction` — the task description (required)
- `input` — extra context, e.g. the text to classify or summarise (optional, leave `""` if unused)
- `response` — the expected output (required)

Save it anywhere, e.g. `data/my_dataset.json`.

---

## 2. Format and split

```bash
uv run python scripts/prepare_data.py \
    --raw-path data/my_dataset.json \
    --format phi3 \
    --output-dir data/processed
```

- `--format phi3` wraps examples in the Phi-3 ChatML template; use `alpaca` for other models
- Outputs `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl` with an 80/10/10 split by default

---

## 3. Fine-tune

### Linux + GPU

```bash
uv sync --group train
uv run --group train python scripts/train.py \
    --model-id microsoft/Phi-3-mini-4k-instruct \
    --quantization 4bit \
    --epochs 3 \
    --output-dir outputs/my-run \
    --merge
```

### macOS / CPU

```bash
uv sync --group train
uv run --group train python scripts/train.py \
    --local-model-dir models/smollm2-360m \
    --no-unsloth --quantization none \
    --max-steps 10 --batch-size 1 --grad-accum 1 \
    --output-dir outputs/my-run --report-to none --merge
```

`--merge` fuses the LoRA adapter into the base model, producing a standard checkpoint that inference scripts can load directly.

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--model-id` | `microsoft/Phi-3-mini-4k-instruct` | Any HF model ID |
| `--local-model-dir` | `None` | Use a local model directory instead |
| `--quantization` | `4bit` | `4bit` / `8bit` / `none` |
| `--lora-r` | `16` | LoRA rank — higher gives more capacity, uses more memory |
| `--epochs` | `3` | Training epochs |
| `--max-steps` | `-1` | Cap total steps (`-1` means run all epochs) |
| `--max-seq-length` | `2048` | Reduce to `512` if you're tight on RAM |
| `--no-unsloth` | off | Required on macOS; disables Unsloth |

---

## 4. Run inference

### macOS / CPU / any platform

```bash
uv run python scripts/inference.py \
    --model-path outputs/my-run/merged \
    --engine transformers \
    --prompt "Your prompt here"
```

### Linux + GPU (vLLM)

```bash
uv sync --group inference
uv run --group inference python scripts/inference.py \
    --model-path outputs/my-run/merged \
    --engine vllm \
    --prompt "Your prompt here"
```

### Batch (one prompt per line in a text file)

```bash
uv run python scripts/inference.py \
    --engine transformers \
    --model-path outputs/my-run/merged \
    --batch-file prompts.txt
```

---

## Switching models

The project works with any causal LM on HuggingFace. Steps:

1. Change `--model-id` (or download locally first with `scripts/download_model.py`)
2. If the model uses a different chat format, use `--format alpaca` or add a formatter in `src/phi3_sft/data/formatter.py`
3. Adjust `--lora-r` and `--max-seq-length` to match your available memory

---

## Platform notes

| | macOS / CPU | Linux + CUDA |
|---|---|---|
| Training | yes (use `--no-unsloth --quantization none`) | yes (Unsloth + 4-bit is fastest) |
| Transformers inference | yes | yes |
| vLLM inference | no | yes |
