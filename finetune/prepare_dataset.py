#!/usr/bin/env python3
"""
Step 2 — Chunk extracted text into training sequences and split train / val.

Output is two JSONL files (one "text" key per line) ready for SFTTrainer.

Chunk-size guidance (completion-style / book-writing):
    ~2000 chars  ≈  500 tokens  →  fits one 512-token sequence
    ~4000 chars  ≈ 1000 tokens  →  use with --max-seq-len 1024 in finetune.py

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --input-dir ./extracted_text --max-samples 50000
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def chunk_text(text, chunk_size=2000, overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if len(piece) > 100:          # skip tiny remnants
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning dataset")
    parser.add_argument("--input-dir",   default="./extracted_text",
                        help="Directory of .txt files (default: ./extracted_text)")
    parser.add_argument("--output-dir",  default="./dataset",
                        help="Output directory (default: ./dataset)")
    parser.add_argument("--chunk-size",  type=int, default=2000,
                        help="Characters per chunk (default: 2000 ≈ 512 tokens)")
    parser.add_argument("--overlap",     type=int, default=200,
                        help="Overlap between chunks (default: 200)")
    parser.add_argument("--val-ratio",   type=float, default=0.05,
                        help="Fraction held out for validation (default: 0.05)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap total samples — handy for a quick test run")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load ---
    txt_files = sorted(input_dir.glob("*.txt"))
    print(f"\n📂 {input_dir}  →  {len(txt_files)} text files")

    # --- chunk ---
    print(f"✂️  Chunking  (size={args.chunk_size}, overlap={args.overlap})…")
    all_chunks = []
    for f in tqdm(txt_files, desc="Chunking"):
        try:
            text = f.read_text(encoding="utf-8")
            all_chunks.extend(chunk_text(text, args.chunk_size, args.overlap))
        except Exception as e:
            print(f"\n⚠️  {f.name}: {e}")

    print(f"✅ Total chunks: {len(all_chunks):,}")

    # --- shuffle ---
    random.seed(args.seed)
    random.shuffle(all_chunks)

    # --- optional cap ---
    if args.max_samples and args.max_samples < len(all_chunks):
        all_chunks = all_chunks[:args.max_samples]
        print(f"🎯 Capped at {args.max_samples:,} samples")

    # --- split ---
    val_n     = max(1, int(len(all_chunks) * args.val_ratio))
    val_data  = all_chunks[:val_n]
    train_data = all_chunks[val_n:]

    # --- time estimate (rough) ---
    eff_batch   = 4                        # default batch 1 × grad_accum 4
    steps       = len(train_data) // eff_batch
    secs_per_step = 0.45                   # empirical on RTX 3070, seq_len 512
    hours       = steps * secs_per_step / 3600

    print(f"\n📊 Train : {len(train_data):>8,}  samples")
    print(f"   Val   : {len(val_data):>8,}  samples")
    print(f"   Steps / epoch  : {steps:,}")
    print(f"   Est. time / epoch (3070) : {hours:.1f} h")

    # --- write JSONL ---
    train_path = output_dir / "train.jsonl"
    val_path   = output_dir / "val.jsonl"

    with open(train_path, "w") as fh:
        for chunk in train_data:
            fh.write(json.dumps({"text": chunk}) + "\n")

    with open(val_path, "w") as fh:
        for chunk in val_data:
            fh.write(json.dumps({"text": chunk}) + "\n")

    print(f"\n💾 {train_path}")
    print(f"💾 {val_path}")
    print(f"\n  Next step:  python finetune.py")


if __name__ == "__main__":
    main()
