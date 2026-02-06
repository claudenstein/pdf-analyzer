#!/bin/bash
# Helper script to download trained model from cloud GPU instance

if [ "$#" -ne 2 ]; then
    echo "Usage: ./download_model.sh <user@host> <port>"
    echo ""
    echo "Example (from vast.ai 'Connect' button):"
    echo "  ssh -p 12345 root@123.45.67.89"
    echo "  → run: ./download_model.sh root@123.45.67.89 12345"
    echo ""
    echo "This downloads lora_output/final/ (the trained adapter)."
    echo "To download a specific checkpoint instead:"
    echo "  rsync -avz -e 'ssh -p 12345' root@123.45.67.89:~/pdf-analyzer/finetune/lora_output/checkpoint-1200/ ./lora_output/"
    exit 1
fi

HOST=$1
PORT=$2

echo "Downloading lora_output/final/ from $HOST..."
echo ""

mkdir -p ./lora_output

rsync -avz --progress -e "ssh -p $PORT" \
  $HOST:~/pdf-analyzer/finetune/lora_output/final/ \
  ./lora_output/final/

echo ""
echo "✅ Download complete!"
echo ""
echo "Next step:"
echo "  python3 export_model.py"
