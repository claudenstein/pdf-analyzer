#!/bin/bash
# Multi-GPU training launcher for QLoRA fine-tuning
# Uses all available GPUs via Hugging Face Accelerate

set -e

echo "=========================================="
echo "  Multi-GPU QLoRA Training Launcher"
echo "=========================================="

# Detect GPU count
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")

if [ "$NUM_GPUS" -eq "0" ]; then
    echo "❌ No CUDA GPUs detected!"
    exit 1
fi

echo "📊 Detected $NUM_GPUS GPU(s)"
echo ""

# Generate accelerate config on the fly
cat > /tmp/accelerate_config.yaml <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "🚀 Launching training on $NUM_GPUS GPUs..."
echo ""

# Launch with accelerate
accelerate launch \
    --config_file /tmp/accelerate_config.yaml \
    finetune.py "$@"

# Clean up
rm -f /tmp/accelerate_config.yaml
