# Cloud Fine-Tuning Guide

Running the fine-tuning job on a cloud GPU cuts training time from **1 week locally** down to **4–12 hours** for $1–3.

> **🚀 First time using vast.ai?** See **[VAST_QUICKSTART.md](VAST_QUICKSTART.md)** for a complete step-by-step walkthrough with every single click and command explained.

---

## Cost Comparison (2026 pricing)

| Provider | GPU | VRAM | $/hr | Est. time (2 epochs, 30k samples) | Total cost |
|---|---|---|---|---|---|
| **vast.ai** | RTX 3090 | 24 GB | ~$0.13 | 8–12 hrs | **$1–2** |
| **vast.ai** | RTX 4090 | 24 GB | ~$0.29 | 4–6 hrs | **$1–2** |
| **RunPod** | RTX 3090 | 24 GB | ~$0.25 | 8–12 hrs | $2–3 |
| Lambda Labs | A100 (40GB) | 40 GB | ~$1.10 | 2–3 hrs | $2–3 |
| Google Colab Pro+ | A100 | 40 GB | $50/mo | 2–3 hrs | (subscription) |

**Recommendation:** Use **vast.ai with an RTX 3090** — best price/performance for 7B models.

---

## Step-by-Step: vast.ai

### 1. Create an account

Go to [vast.ai](https://vast.ai) and sign up. Add $5–10 credit (PayPal/card).

---

### 2. Launch an instance

1. Click **"Rent"** at the top
2. Set filters:
   - **GPU**: RTX 3090, RTX 4090, or A5000
   - **VRAM**: ≥ 20 GB
   - **Disk**: ≥ 50 GB (for model cache + outputs)
   - **DLPerf**: ≥ 50 (network speed — important for model download)
3. **Template**: Select **"pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"** or any Ubuntu 22.04 + CUDA 12.x image
4. **On-start script**: Leave blank (we'll run `cloud_setup.sh` manually)
5. Sort by **$/hr** and pick the cheapest that meets the filters
6. Click **"Rent"** — the instance boots in ~30 seconds

---

### 3. Connect via SSH

Once the instance shows "running", click **"Connect"** and copy the SSH command. It looks like:

```bash
ssh -p 12345 root@123.45.67.89 -L 8080:localhost:8080
```

Run that from your local terminal. You're now inside the cloud GPU.

---

### 4. Set up the environment

```bash
# Clone the repo (or upload finetune/ directly — see next section)
git clone https://github.com/claudenstein/pdf-analyzer.git
cd pdf-analyzer/finetune

# Run the setup script
chmod +x cloud_setup.sh
./cloud_setup.sh
```

This installs PyTorch, transformers, bitsandbytes, trl, etc. Takes ~5 minutes.

---

### 5. Upload your dataset

From your **local machine** (not the cloud instance), run:

```bash
# Find the instance IP and port from the vast.ai "Connect" button
# Example: ssh -p 12345 root@123.45.67.89

# Upload the dataset folder
rsync -avz -e "ssh -p 12345" \
  ~/path/to/pdf-analyzer/finetune/dataset/ \
  root@123.45.67.89:~/pdf-analyzer/finetune/dataset/
```

Replace `12345` and `123.45.67.89` with your actual port and IP. This uploads `train.jsonl` and `val.jsonl`.

**Alternative (if rsync isn't available):** use `scp`:

```bash
scp -P 12345 -r ~/path/to/pdf-analyzer/finetune/dataset \
  root@123.45.67.89:~/pdf-analyzer/finetune/
```

---

### 6. Run the fine-tuning job

Back inside the **cloud SSH session**:

```bash
cd ~/pdf-analyzer/finetune

# Verify the dataset uploaded correctly
ls -lh dataset/
# You should see train.jsonl and val.jsonl

# Start training
python3 finetune.py
```

The first run downloads the base model (~14 GB, takes 5–10 min depending on DLPerf). Subsequent runs use the cache.

**Monitor progress:** The script prints loss every 25 steps and saves checkpoints every 200 steps to `./lora_output/`.

---

### 7. Download the trained model

Once training finishes (you'll see `💾 Saving LoRA weights → ./lora_output/final`), download the result from your **local machine**:

```bash
# Download the final LoRA adapter
rsync -avz -e "ssh -p 12345" \
  root@123.45.67.89:~/pdf-analyzer/finetune/lora_output/final/ \
  ~/pdf-analyzer/finetune/lora_output/final/

# Or download a specific checkpoint
rsync -avz -e "ssh -p 12345" \
  root@123.45.67.89:~/pdf-analyzer/finetune/lora_output/checkpoint-1200/ \
  ~/pdf-analyzer/finetune/lora_output/checkpoint-1200/
```

---

### 8. Destroy the instance

In the vast.ai web console, click **"Destroy"** next to your instance. Billing stops immediately. The instance and all data on it are deleted.

**Important:** Download your `lora_output/` folder BEFORE destroying the instance — once it's gone, the data is unrecoverable.

---

## Troubleshooting

### "CUDA out of memory"

Your GPU ran out of VRAM. Try in order:

1. **Reduce sequence length:**
   ```bash
   python3 finetune.py --max-seq-len 256
   ```

2. **Reduce LoRA rank:**
   ```bash
   python3 finetune.py --max-seq-len 256 --lora-rank 8
   ```

3. **Rent a bigger GPU** (e.g., RTX 4090 or A5000 with 24 GB instead of 3090).

### "Model download is stuck"

Some vast.ai hosts have slow network. Check the **DLPerf** score when renting — anything below 50 is risky. Destroy and rent a different instance with DLPerf ≥ 100.

### "Can't connect via SSH"

1. Check the instance status is "running" (not "loading")
2. Copy the exact SSH command from the vast.ai "Connect" button — the port changes every time
3. If it still fails, the host may have firewall issues — destroy and rent a different one

---

## Alternative: RunPod

If vast.ai availability is low, [RunPod](https://runpod.io) is the next best option:

1. Sign up at runpod.io
2. Click **"Deploy"** → **"GPU Instances"**
3. Select **"pytorch"** template
4. Choose RTX 3090 or 4090
5. Click **"Deploy On-Demand"**
6. Connect via the web-based Jupyter terminal or SSH (shown in the console)
7. Run the same `cloud_setup.sh` and `finetune.py` commands

RunPod is slightly more expensive (~$0.25/hr for 3090) but has better uptime and a friendlier UI.

---

## Next Steps After Fine-Tuning

Once you've downloaded `lora_output/final/`, proceed to **Step 4** in the main `finetune/README.md`:

```bash
# Merge LoRA weights and export to GGUF for Ollama
python3 export_model.py
```

This runs locally (CPU-only, no GPU needed) and takes ~10 minutes. You'll get a quantized GGUF file ready for `ollama create`.

---

## Sources

- [vast.ai pricing](https://vast.ai/pricing)
- [GPU rental comparison (2026)](https://computeprices.com/providers/vast)
- [Cheapest cloud GPU providers](https://northflank.com/blog/cheapest-cloud-gpu-providers)
