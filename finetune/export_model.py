#!/usr/bin/env python3
"""
Step 4 — Merge LoRA weights, convert to GGUF, register with Ollama.

Pipeline:
    1. Load the base model on CPU (~14 GB RAM) and the LoRA adapter.
    2. Merge adapter weights into the base weights.
    3. Clone llama.cpp (if needed) and run its convert_hf_to_gguf.py.
    4. Write a Modelfile so `ollama create` can finish the job.

64 GB RAM is plenty for every step here.

Usage:
    python export_model.py
    python export_model.py --lora-path ./lora_output/final --ollama-name my-science-model
    python export_model.py --skip-merge --skip-convert        # regenerate Modelfile only
"""

import gc
import sys
import argparse
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ---------------------------------------------------------------------------
# Step 1 — Merge
# ---------------------------------------------------------------------------

def merge_lora(base_model_name, lora_path, output_path):
    """Load base + LoRA on CPU, merge, save as safetensors."""
    print("\n📥 Loading base model on CPU  (~14 GB RAM) …")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print("🔗 Loading LoRA adapter …")
    model = PeftModel.from_pretrained(base, lora_path)

    print("🔀 Merging weights …")
    model = model.merge_and_unload()

    print(f"💾 Saving merged model → {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)

    # free the ~14 GB we just used
    del model, base
    gc.collect()
    print("✅ Merge complete\n")


# ---------------------------------------------------------------------------
# Step 2 — GGUF conversion via llama.cpp
# ---------------------------------------------------------------------------

def ensure_llama_cpp():
    """Clone llama.cpp into CWD if not already present."""
    dest = Path("./llama.cpp")
    script = dest / "convert_hf_to_gguf.py"
    if script.exists():
        print("✅ llama.cpp already present")
        return dest

    print("📥 Cloning llama.cpp (shallow) …")
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/ggerganov/llama.cpp.git",
        str(dest),
    ], check=True)

    # install its Python deps
    reqs = dest / "requirements-convert-hf-to-gguf.txt"
    if reqs.exists():
        print("📦 Installing llama.cpp conversion requirements …")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(reqs)],
            check=True,
        )
    else:
        # minimal fallback set
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "gguf", "numpy", "safetensors", "sentencepiece"],
            check=True,
        )
    return dest


def convert_to_gguf(merged_path, gguf_path, quant="Q4_K_M"):
    """Run llama.cpp's convert_hf_to_gguf.py."""
    llama_cpp  = ensure_llama_cpp()
    script     = llama_cpp / "convert_hf_to_gguf.py"

    print(f"🔄 Converting to GGUF  (quant={quant}) …")
    print(f"   Model dir : {merged_path}")
    print(f"   Output    : {gguf_path}\n")

    result = subprocess.run([
        sys.executable,
        str(script),
        str(merged_path),
        "--outfile", str(gguf_path),
        "--outtype", quant.lower(),
    ])

    if result.returncode != 0:
        print("\n❌ Conversion exited with an error.")
        print("   Try running the command manually:")
        print(f"   python llama.cpp/convert_hf_to_gguf.py "
              f"{merged_path} --outfile {gguf_path} --outtype {quant.lower()}")
        return False

    size_gb = Path(gguf_path).stat().st_size / 1e9
    print(f"\n✅ GGUF written  →  {gguf_path}  ({size_gb:.2f} GB)")
    return True


# ---------------------------------------------------------------------------
# Step 3 — Ollama Modelfile
# ---------------------------------------------------------------------------

def create_modelfile(gguf_path, model_name):
    """Write an Ollama Modelfile and print next steps."""
    gguf_abs = Path(gguf_path).resolve()

    content = (
        f"FROM {gguf_abs}\n"
        "\n"
        "# Sampling parameters\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p        0.9\n"
        "PARAMETER top_k        50\n"
        "PARAMETER num_ctx      4096\n"
        "\n"
        "# System prompt\n"
        'SYSTEM You are an expert scientific writer. '
        'You synthesise knowledge from research papers and literature '
        'to produce clear, well-structured, and accurate long-form writing.\n'
    )

    modelfile_path = Path("Modelfile")
    modelfile_path.write_text(content)

    print(f"\n📄 Modelfile  →  {modelfile_path}")
    print(f"\n   Register with Ollama:")
    print(f"     ollama create {model_name} -f Modelfile")
    print(f"\n   Use interactively:")
    print(f"     ollama run {model_name}")
    print(f"\n   Or plug into pdf_analyzer_pro:")
    print(f"     python ../pdf_analyzer_pro.py --model {model_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model → Ollama")

    parser.add_argument("--base-model",   default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--lora-path",    default="./lora_output/final",
                        help="LoRA adapter directory (default: ./lora_output/final)")
    parser.add_argument("--merged-dir",   default="./merged_model",
                        help="Where to save the merged model (default: ./merged_model)")
    parser.add_argument("--gguf-path",    default="./scientific-mistral.gguf",
                        help="Output GGUF file (default: ./scientific-mistral.gguf)")
    parser.add_argument("--quant",        default="Q4_K_M",
                        help="GGUF quantisation (default: Q4_K_M  ≈ 4.5 GB)")
    parser.add_argument("--ollama-name",  default="scientific-mistral",
                        help="Name registered in Ollama (default: scientific-mistral)")

    parser.add_argument("--skip-merge",   action="store_true")
    parser.add_argument("--skip-convert", action="store_true")

    args = parser.parse_args()

    print(f"\n{'=' * 58}")
    print(f"  Export Fine-Tuned Model  →  Ollama")
    print(f"{'=' * 58}")
    print(f"  LoRA path     {args.lora_path}")
    print(f"  Merged dir    {args.merged_dir}")
    print(f"  GGUF output   {args.gguf_path}")
    print(f"  Quantisation  {args.quant}")
    print(f"  Ollama name   {args.ollama_name}")
    print(f"{'=' * 58}\n")

    # 1 — merge
    if args.skip_merge:
        print("⏭️  Skipping merge")
    else:
        merge_lora(args.base_model, args.lora_path, args.merged_dir)

    # 2 — GGUF
    if args.skip_convert:
        print("⏭️  Skipping GGUF conversion")
    else:
        if not convert_to_gguf(args.merged_dir, args.gguf_path, args.quant):
            print("\n⚠️  GGUF conversion failed — see manual instructions above.")

    # 3 — Modelfile
    create_modelfile(args.gguf_path, args.ollama_name)

    print(f"\n{'=' * 58}")
    print(f"  ✅ Export complete")
    print(f"{'=' * 58}\n")


if __name__ == "__main__":
    main()
