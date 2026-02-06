#!/bin/bash
# Quick setup script for cloud GPU instances (vast.ai, RunPod, Lambda, etc.)
# Run this on a fresh Ubuntu 22.04 + CUDA 12.x instance

set -e

echo "=========================================="
echo "  Cloud GPU Setup for QLoRA Fine-Tuning"
echo "=========================================="

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y git wget curl python3-pip

# Verify CUDA
echo ""
echo "🔍 Checking CUDA installation..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found! Make sure you selected a CUDA template."
    exit 1
fi
nvidia-smi

# Install Python deps
echo ""
echo "🐍 Installing Python packages (this takes ~5 min)..."
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install transformers>=4.36 datasets>=2.18 accelerate>=0.28
pip3 install bitsandbytes>=0.41 peft>=0.7 trl>=0.7
pip3 install tqdm safetensors

# Verify GPU access in PyTorch
echo ""
echo "✅ Verifying PyTorch + CUDA..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('ERROR: CUDA not available in PyTorch')
    exit(1)
"

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Upload your dataset/ folder to this instance"
echo "  2. Run: python3 finetune.py"
echo "=========================================="
