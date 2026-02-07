# vast.ai First-Time Setup — Complete Walkthrough

This guide assumes you've never used vast.ai before and walks through every single step to fine-tune your model on a cloud GPU for ~$1–2.

**Total time:** ~30 minutes of setup, then 8–12 hours of unattended training.

---

## Part 1: Create Account & Add Credit (5 minutes)

### Step 1.1: Sign up

1. Open your browser and go to **https://vast.ai**
2. Click the **"Sign Up"** button (top right corner)
3. Enter your email and create a password
4. Check your email and click the verification link
5. Log back into vast.ai

### Step 1.2: Add billing credit

1. Once logged in, click your **username** (top right) → **"Billing"**
2. Click **"Add Credit"**
3. Choose an amount: **$5** is enough for a test run, **$10** covers the full job + safety margin
4. Select payment method (PayPal or credit card)
5. Complete the payment
6. You should see your balance update (e.g., "$10.00" in the top right)

---

## Part 2: Find and Rent a GPU (10 minutes)

### Step 2.1: Open the GPU search page

1. Click **"Search"** in the top menu (or just go to https://cloud.vast.ai/)
2. You'll see a big list of available GPU instances with prices

### Step 2.2: Set filters to find the right GPU

On the left sidebar, set these filters:

1. **GPU Name:** Click the dropdown and check:
   - ☑ RTX 3090
   - ☑ RTX 4090
   - ☑ RTX A5000

2. **VRAM (GB):** Set the slider to **≥ 20 GB**

3. **Disk Space (GB):** Set to **≥ 50 GB**

4. **DLPerf:** Set to **≥ 50** (this is network speed — important for downloading the 14 GB model)

5. **Verified:** Toggle this ON (avoids unreliable hosts)

6. Leave everything else at default

### Step 2.3: Sort by price

At the top of the results table, click the **"$/hr"** column header to sort from cheapest to most expensive.

You should now see RTX 3090s at around **$0.13–0.20/hr** at the top of the list.

### Step 2.4: Pick an instance

Look at the first few rows. Each row shows:
- **GPU model** (e.g., "RTX 3090")
- **$/hr** (e.g., "$0.13")
- **DLPerf** (e.g., "120" — higher is better)
- **Reliability** (a percentage — aim for ≥95%)

Pick one that looks good (cheap + high DLPerf + reliable). Click the **"Rent"** button on that row.

### Step 2.5: Configure the instance

A popup appears with configuration options:

1. **Image & Config:**
   - Click **"Select"** next to **"pytorch/pytorch"** (look for something like `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`)
   - If you can't find that exact one, pick any image that says **"Ubuntu 22.04"** + **"CUDA 12.x"**

2. **Disk Space:**
   - Set to **50 GB** (default is usually fine)

3. **On-start script:**
   - Leave this **blank** (we'll run setup manually)

4. **SSH:**
   - The checkbox **"Use SSH"** should be ON (it is by default)
   - If you already have an SSH key uploaded, it will use that. If not, vast.ai will generate a password for you.

5. Click the big blue **"Rent"** button at the bottom

### Step 2.6: Wait for the instance to start

You'll be taken to the **"Instances"** page. Your new instance will show:
- **Status:** "loading…" (turns to "running" after ~30–60 seconds)

Wait until the status shows **"running"**.

---

## Part 3: Connect to Your Cloud GPU (5 minutes)

### Step 3.1: Get the SSH command

Once the instance is **"running"**:

1. Click the **"Connect"** button next to your instance
2. A popup shows an SSH command that looks like:
   ```
   ssh -p 41234 root@12.34.56.78 -L 8080:localhost:8080
   ```
3. Click the **"Copy"** button to copy this command

### Step 3.2: Open a terminal on your local machine

- **Linux/Mac:** Open the "Terminal" app
- **Windows:** Open "PowerShell" or install [Git Bash](https://git-scm.com/downloads) if you don't have SSH

### Step 3.3: Connect

Paste the SSH command you copied and press Enter:

```bash
ssh -p 41234 root@12.34.56.78 -L 8080:localhost:8080
```

The first time you connect, you'll see:
```
The authenticity of host '[12.34.56.78]:41234' can't be established.
Are you sure you want to continue connecting (yes/no)?
```

Type **`yes`** and press Enter.

If vast.ai generated a password for you, it will show in the "Connect" popup — copy and paste it when prompted.

**You're now inside the cloud GPU!** The prompt will change to something like:
```
root@vast-instance:~#
```

---

## Part 4: Set Up the Environment (5 minutes)

### Step 4.1: Clone the repository

Inside the SSH session, run:

```bash
git clone https://github.com/claudenstein/pdf-analyzer.git
cd pdf-analyzer/finetune
```

### Step 4.2: Run the setup script

```bash
chmod +x cloud_setup.sh
./cloud_setup.sh
```

This will:
- Update system packages
- Install PyTorch with CUDA support
- Install transformers, bitsandbytes, peft, trl, etc.

It takes **~5 minutes**. You'll see lots of output scrolling by — that's normal.

At the end, you should see:
```
✅ Setup complete!
GPU detected: NVIDIA GeForce RTX 3090
VRAM: 24.0 GB
```

If you see that, the GPU is working correctly.

---

## Part 5: Upload Your Dataset (3 minutes)

### Step 5.1: Open a NEW terminal on your local machine

**Don't close the SSH session!** Open a second terminal window or tab. This one stays on your local machine.

### Step 5.2: Navigate to your local pdf-analyzer folder

```bash
cd ~/path/to/pdf-analyzer/finetune
# Replace ~/path/to/ with wherever you cloned the repo
```

### Step 5.3: Upload the dataset

Run the upload helper script:

```bash
./upload_dataset.sh root@12.34.56.78 41234
```

**Important:** Replace `12.34.56.78` and `41234` with YOUR instance's IP and port (from the SSH command you copied earlier).

This uploads your `train.jsonl` and `val.jsonl` files. You'll see a progress bar:

```
sending incremental file list
dataset/train.jsonl
     12,345,678 100%   10.5MB/s    0:00:01
dataset/val.jsonl
        654,321 100%    8.2MB/s    0:00:00
```

When it finishes:
```
✅ Upload complete!
```

---

## Part 6: Start Training (1 minute)

### Step 6.1: Go back to the SSH terminal

Switch back to the terminal window where you're connected to the cloud GPU.

### Step 6.2: Verify the dataset uploaded

```bash
ls -lh dataset/
```

You should see:
```
train.jsonl
val.jsonl
```

### Step 6.3: Start the fine-tuning job

**If you have multiple GPUs (e.g., 2× RTX 3090):**
```bash
chmod +x train_multi_gpu.sh
./train_multi_gpu.sh
```

**If you have a single GPU:**
```bash
python3 finetune.py
```

**If you have a GPU with >20 GB VRAM, increase batch size for speed:**
```bash
python3 finetune.py --batch-size 4
```

You'll see:
```
==========================================================
  QLoRA Fine-Tuning — Mistral 7B
==========================================================
  GPUs           2× NVIDIA GeForce RTX 3090  (24.0 GB each)
  Model          mistralai/Mistral-7B-v0.1
  ...
```

**First run only:** The script downloads the base model (~14 GB). This takes 5–15 minutes depending on the host's network speed.

Then training starts. You'll see progress like:
```
{'loss': 2.1234, 'learning_rate': 0.0002, 'epoch': 0.05}
{'loss': 1.8567, 'learning_rate': 0.00019, 'epoch': 0.10}
...
```

**Training time:**
- Single RTX 3090: ~8–12 hours
- 2× RTX 3090 with multi-GPU script: ~4–6 hours
- Single RTX 3090 with `--batch-size 4`: ~5–8 hours

You can now:
- **Close the terminal** (training keeps running in the background)
- Or **keep it open** if you want to watch progress

To safely disconnect while keeping training running, press **`Ctrl+A`** then **`D`** if you're in a `screen` session, OR just close the terminal — the process stays alive.

**To reconnect later and check progress:**
1. Go back to vast.ai → "Instances" → "Connect"
2. SSH back in
3. Run:
   ```bash
   cd pdf-analyzer/finetune
   tail -f lora_output/trainer_state.json
   ```
   This shows live training logs.

---

## Part 7: Download the Trained Model (5 minutes)

### Step 7.1: Wait for training to finish

Training is done when you see:
```
💾 Saving LoRA weights → ./lora_output/final
```

Or check the vast.ai console — if the instance has been running for ~8–12 hours, it's probably done.

### Step 7.2: Download the model to your local machine

From your **local terminal** (not the SSH session), run:

```bash
cd ~/path/to/pdf-analyzer/finetune
./download_model.sh root@12.34.56.78 41234
```

Again, replace the IP and port with your instance's values.

This downloads the trained LoRA weights to your local machine. You'll see:
```
receiving incremental file list
lora_output/final/adapter_config.json
lora_output/final/adapter_model.safetensors
...
✅ Download complete!
```

The trained model is now in `./lora_output/final/` on your local machine.

---

## Part 8: Destroy the Instance (1 minute)

**⚠️ IMPORTANT:** Once you've downloaded the model, destroy the instance immediately or you keep getting charged!

1. Go to https://cloud.vast.ai/ (the "Instances" page)
2. Find your running instance
3. Click the **"Destroy"** button (it's red, on the right side of the row)
4. Confirm by clicking **"Yes, destroy"**

The instance shuts down within seconds and billing stops. All data on the instance is deleted (which is why you downloaded `lora_output/` first).

Check your balance in the top right — it should show how much credit you have left.

---

## Part 9: Export and Use the Model (10 minutes, runs locally)

Now that you have the trained LoRA adapter, merge it and export to Ollama:

```bash
cd ~/path/to/pdf-analyzer/finetune

# This runs on your local machine (CPU-only, no GPU needed)
python3 export_model.py
```

This takes ~10 minutes and produces a GGUF file ready for Ollama.

At the end, you'll see:
```
🎉 All done!  Run:  ollama create scientific-mistral -f Modelfile
```

Run that command and your fine-tuned model is ready to use:

```bash
ollama create scientific-mistral -f Modelfile
ollama run scientific-mistral
```

---

## Total Cost Breakdown

| Item | Cost |
|---|---|
| RTX 3090 @ $0.13/hr × 10 hours | **$1.30** |
| Buffer for slower hosts or reruns | +$0.70 |
| **Total** | **~$2.00** |

You'll have $8 left from your initial $10 credit for future runs.

---

## Troubleshooting

### "ssh: connect to host 12.34.56.78 port 41234: Connection refused"

The instance isn't ready yet. Wait 30 more seconds and try again. If it still fails after 2 minutes, the host may be down — destroy the instance and rent a different one.

### "CUDA out of memory" during training

The GPU ran out of VRAM. SSH back in and restart with smaller settings:

```bash
python3 finetune.py --max-seq-len 256
```

If it still OOMs:
```bash
python3 finetune.py --max-seq-len 256 --lora-rank 8
```

### Model download is stuck at 0%

The host has slow internet. Check the **DLPerf** score in the instance list — if it's below 50, destroy and rent one with DLPerf ≥ 100.

### I closed the terminal and lost the SSH connection — how do I check if training is still running?

1. Reconnect via SSH (use the same "Connect" command from vast.ai)
2. Run:
   ```bash
   cd pdf-analyzer/finetune
   ls lora_output/
   ```
   If you see checkpoint folders like `checkpoint-200`, `checkpoint-400`, training is progressing.
3. To watch live:
   ```bash
   tail -f lora_output/runs/*/events.out.tfevents.*
   ```
   Or just wait and check back in a few hours.

### I forgot to download the model before destroying the instance!

The data is gone. You'll need to rent a new instance and re-run training. Always download `lora_output/` before clicking "Destroy".

---

## Quick Reference — Commands Cheat Sheet

**On the cloud GPU (via SSH):**
```bash
git clone https://github.com/claudenstein/pdf-analyzer.git
cd pdf-analyzer/finetune
./cloud_setup.sh
python3 finetune.py
```

**On your local machine:**
```bash
cd ~/path/to/pdf-analyzer/finetune
./upload_dataset.sh root@IP PORT
./download_model.sh root@IP PORT
python3 export_model.py
ollama create scientific-mistral -f Modelfile
```

**Check training progress (on cloud GPU):**
```bash
cd pdf-analyzer/finetune
ls lora_output/    # see checkpoint folders
tail lora_output/trainer_state.json    # see last logged step
```

---

## Next Steps

After you've done this once, you'll find it takes less than 5 minutes to launch the next training run — you'll know exactly which buttons to click and commands to run.

If you want to tweak the training (try different learning rates, more epochs, etc.), just edit the command:

```bash
# Try a lower learning rate
python3 finetune.py --lr 0.0001

# Train for 4 epochs instead of 2
python3 finetune.py --epochs 4

# Quick smoke test (finishes in ~30 min)
python3 finetune.py --epochs 1 --max-seq-len 256
```

All available flags are documented in `finetune.py --help`.

Good luck with your fine-tuning! 🚀
