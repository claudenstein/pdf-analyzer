#!/usr/bin/env python3
"""
Step 3 — QLoRA fine-tuning of Mistral 7B.

Tuned for an RTX 3070 (8 GB VRAM):
    • 4-bit NF4 quantisation  (bitsandbytes)
    • LoRA on attention + MLP (peft)
    • 8-bit paged optimiser   (saves ~2 GB vs Adam fp32)
    • Gradient checkpointing  (trades speed for memory)
    • Sequence packing        (fills every token in a batch)

If you get an OOM, try in order:
    1.  --max-seq-len 256
    2.  --lora-rank 8
    3.  Remove gate_proj / up_proj / down_proj from target_modules below

Usage:
    # Quick smoke-test (~5 min)
    python finetune.py --epochs 1 --max-seq-len 256

    # Full run
    python finetune.py --epochs 2

    # Resume an interrupted run
    python finetune.py --resume ./lora_output/checkpoint-400
"""

import sys
import json
import argparse

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """Read a JSONL file into a HuggingFace Dataset."""
    rows = []
    with open(path) as fh:
        for line in fh:
            rows.append(json.loads(line))
    return Dataset.from_dict({"text": [r["text"] for r in rows]})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Mistral 7B")

    # --- model / data ---
    parser.add_argument("--model",      default="mistralai/Mistral-7B-v0.1",
                        help="HuggingFace model ID")
    parser.add_argument("--train-file", default="./dataset/train.jsonl")
    parser.add_argument("--val-file",   default="./dataset/val.jsonl")
    parser.add_argument("--output-dir", default="./lora_output")

    # --- training ---
    parser.add_argument("--epochs",     type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int,   default=1,
                        help="Per-device batch size (keep 1 for 8 GB VRAM)")
    parser.add_argument("--grad-accum", type=int,   default=4,
                        help="Effective batch = batch-size × grad-accum")
    parser.add_argument("--max-seq-len",type=int,   default=512,
                        help="Tokens per packed sequence (512 safe on 3070; try 1024 if headroom)")

    # --- LoRA ---
    parser.add_argument("--lora-rank",  type=int,   default=16)
    parser.add_argument("--lora-alpha", type=int,   default=32)

    # --- resume ---
    parser.add_argument("--resume",     default=None,
                        help="Path to a checkpoint directory to resume from")

    args = parser.parse_args()

    # ── GPU check ──────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9

    # ── banner ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 58}")
    print(f"  QLoRA Fine-Tuning — Mistral 7B")
    print(f"{'=' * 58}")
    print(f"  GPU            {gpu_name}  ({vram_gb:.1f} GB)")
    print(f"  Model          {args.model}")
    print(f"  Train file     {args.train_file}")
    print(f"  Val file       {args.val_file}")
    print(f"  Output         {args.output_dir}")
    print(f"  Epochs         {args.epochs}")
    print(f"  LR             {args.lr}")
    print(f"  Eff. batch     {args.batch_size * args.grad_accum}")
    print(f"  Seq length     {args.max_seq_len} tokens")
    print(f"  LoRA rank / α  {args.lora_rank} / {args.lora_alpha}")
    print(f"{'=' * 58}\n")

    # ── datasets ───────────────────────────────────────────────────────────
    print("📂 Loading datasets …")
    train_ds = load_jsonl(args.train_file)
    val_ds   = load_jsonl(args.val_file)
    print(f"   Train : {len(train_ds):,}  |  Val : {len(val_ds):,}\n")

    # ── 4-bit quantisation ─────────────────────────────────────────────────
    print("⚙️  Configuring 4-bit NF4 quantisation …")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── load model ─────────────────────────────────────────────────────────
    print("📥 Loading model …  (first time downloads ~14 GB)\n")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # ── tokeniser ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── prepare for LoRA ───────────────────────────────────────────────────
    print("🔧 Preparing model for QLoRA …")
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        # attention + MLP — maximises learning signal
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── training arguments ─────────────────────────────────────────────────
    train_args = TrainingArguments(
        output_dir=args.output_dir,

        # --- core ---
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,

        # --- optimiser ---
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",

        # --- precision ---
        fp16=False,
        bf16=True,                        # 3070 supports bf16

        # --- logging / checkpoints ---
        logging_steps=25,
        save_steps=200,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=200,
        load_best_model_at_end=True,

        # --- misc ---
        dataloader_pin_memory=False,      # avoids CUDA errors on laptops
        report_to="none",

        # --- resume ---
        resume_from_checkpoint=args.resume,
    )

    # ── trainer ────────────────────────────────────────────────────────────
    print("\n🚀 Starting training …\n")

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        max_seq_length=args.max_seq_len,
        packing=True,                     # concatenates examples → no wasted tokens
    )

    trainer.train(resume_from_checkpoint=args.resume)

    # ── save ───────────────────────────────────────────────────────────────
    final_dir = f"{args.output_dir}/final"
    print(f"\n💾 Saving LoRA weights → {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\n{'=' * 58}")
    print(f"  ✅ Training complete")
    print(f"  Next step :  python export_model.py --lora-path {final_dir}")
    print(f"{'=' * 58}")


if __name__ == "__main__":
    main()
