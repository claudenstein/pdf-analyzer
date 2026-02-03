# Fine-Tuning Pipeline

QLoRA fine-tune Mistral 7B on your scientific papers so it can write long-form
content in that domain.  Designed for an **RTX 3070 (8 GB VRAM)** + 64 GB RAM.

---

## Prerequisites

```bash
# 1. Install deps (from the repo root)
pip install -r finetune/requirements.txt

# 2. Make sure Ollama is running (needed at the end)
ollama serve

# 3. A HuggingFace account — log in once so the model can download
huggingface-cli login
```

---

## The four steps

```
PDF / EPUB files
       │
       ▼
 extract_text.py        →  ./extracted_text/*.txt
       │
       ▼
 prepare_dataset.py     →  ./dataset/train.jsonl  +  val.jsonl
       │
       ▼
 finetune.py            →  ./lora_output/final/   (LoRA weights)
       │
       ▼
 export_model.py        →  scientific-mistral.gguf  +  Modelfile
                            then: ollama create scientific-mistral -f Modelfile
```

---

## Step 1 — Extract text

```bash
cd finetune

# Default: scans ./pdfs recursively, uses CUDA for Docling layout models
python extract_text.py --input-dir ../pdfs

# Multiple directories
python extract_text.py --input-dir ../pdfs ../ebooks

# CPU-only (no GPU required)
python extract_text.py --input-dir ../pdfs --device cpu

# Start fresh — ignore previous progress
python extract_text.py --input-dir ../pdfs --no-resume
```

**Key flags:**

| Flag | Default | What it does |
|------|---------|--------------|
| `--input-dir` | `./pdfs` | One or more directories to scan (PDF + EPUB) |
| `--output-dir` | `./extracted_text` | Where cleaned `.txt` files are written |
| `--min-length` | `200` | Files shorter than this (chars) are skipped |
| `--device` | `cuda` | Inference device for Docling layout models (`cuda`, `cpu`, `auto`) |
| `--no-resume` | — | Ignore previous progress and start from scratch |

**Resume & crash recovery** — progress is saved to `.resume.json` after every file.
If the process crashes (e.g. a segfault in Docling's native backend), it is detected
on the next run and that file is automatically retried with the pdfplumber fallback.
Logs are written to `extract.log` inside the output directory.

---

## Step 2 — Prepare dataset

```bash
python prepare_dataset.py

# Quick test-run dataset (trains in ≈ 1 h on a 3070)
python prepare_dataset.py --max-samples 50000
```

**Key flags:**

| Flag | Default | What it does |
|------|---------|--------------|
| `--input-dir` | `./extracted_text` | Text files produced by Step 1 |
| `--output-dir` | `./dataset` | Where `train.jsonl` / `val.jsonl` land |
| `--chunk-size` | `2000` | Characters per training chunk (≈ 512 tokens) |
| `--overlap` | `200` | Overlap between chunks (keeps cross-paragraph context) |
| `--val-ratio` | `0.05` | 5 % of chunks go to validation |
| `--max-samples` | all | Cap the number of training samples |

The script prints an **estimated training time** before it exits.

---

## Step 3 — Fine-tune (QLoRA)

```bash
# Full run (expect 10-15 h per epoch on a 3070)
python finetune.py

# Quick smoke test — finishes in ~5 min
python finetune.py --epochs 1 --max-seq-len 256

# Resume after a crash or reboot
python finetune.py --resume ./lora_output/checkpoint-400
```

**Key flags:**

| Flag | Default | What it does |
|------|---------|--------------|
| `--model` | `mistralai/Mistral-7B-v0.1` | Base model (first run downloads ≈ 14 GB) |
| `--train-file` | `./dataset/train.jsonl` | Training data |
| `--val-file` | `./dataset/val.jsonl` | Validation data |
| `--output-dir` | `./lora_output` | Checkpoints + final LoRA weights |
| `--epochs` | `2` | Number of passes over the data |
| `--lr` | `2e-4` | Learning rate |
| `--batch-size` | `1` | Per-GPU batch (keep at 1 for 8 GB) |
| `--grad-accum` | `4` | Effective batch = batch-size × this |
| `--max-seq-len` | `512` | Tokens per sequence (try 1024 if no OOM) |
| `--lora-rank` | `16` | LoRA rank — higher = more params, more memory |
| `--lora-alpha` | `32` | LoRA scaling factor (usually 2 × rank) |
| `--resume` | — | Checkpoint path to continue from |

**Out of memory?** Try in this order:

1. `--max-seq-len 256`
2. `--lora-rank 8`
3. Open `finetune.py` and remove `gate_proj / up_proj / down_proj` from `target_modules`

---

## Step 4 — Export to Ollama

```bash
python export_model.py
```

This does three things automatically:

1. **Merges** LoRA weights back into the base model (uses ≈ 14 GB RAM, runs on CPU).
2. **Converts** to GGUF via llama.cpp (cloned for you if missing).
3. **Writes** an Ollama `Modelfile`.

Then finish with:

```bash
ollama create scientific-mistral -f Modelfile
```

**Key flags:**

| Flag | Default | What it does |
|------|---------|--------------|
| `--lora-path` | `./lora_output/final` | LoRA adapter weights |
| `--merged-dir` | `./merged_model` | Intermediate merged model (safetensors) |
| `--gguf-path` | `./scientific-mistral.gguf` | Final GGUF file |
| `--quant` | `Q4_K_M` | GGUF quantisation (~4.5 GB output) |
| `--ollama-name` | `scientific-mistral` | Name used in Ollama |
| `--skip-merge` | — | Skip merging (already done) |
| `--skip-convert` | — | Skip GGUF conversion (already done) |

---

## Using the model

```bash
# Chat directly
ollama run scientific-mistral

# Plug into pdf_analyzer_pro (from repo root)
python pdf_analyzer_pro.py --model scientific-mistral
```

---

## Recommended workflow for 600 files / 3 GB

1. Run Steps 1-2 with **default settings** — takes a few minutes.
2. Do a **smoke test**: `python finetune.py --epochs 1 --max-seq-len 256`.
   Confirm it starts, logs loss, and saves a checkpoint.  ≈ 5 min.
3. Prepare a **limited dataset** for a faster first full run:
   `python prepare_dataset.py --max-samples 100000`
   then `python finetune.py --epochs 1`.  ≈ 3-4 h.
4. Once that model feels right, go for the **full dataset**, 2 epochs.
   ≈ 10-15 h on a 3070.  Leave it overnight.

---

## Laptop tips

* Close other GPU apps (Discord, games, etc.) before training.
* If the laptop shuts down from heat, reduce `--max-seq-len` to 256 or add
  a small delay between steps (fans need time).
* Training can be **resumed** from any checkpoint — no work is lost on a
  crash or reboot.
