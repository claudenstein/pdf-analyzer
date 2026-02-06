#!/bin/bash
# Helper script to upload dataset to a cloud GPU instance

if [ "$#" -ne 2 ]; then
    echo "Usage: ./upload_dataset.sh <user@host> <port>"
    echo ""
    echo "Example (from vast.ai 'Connect' button):"
    echo "  ssh -p 12345 root@123.45.67.89"
    echo "  → run: ./upload_dataset.sh root@123.45.67.89 12345"
    exit 1
fi

HOST=$1
PORT=$2

echo "Uploading dataset/ to $HOST:~/pdf-analyzer/finetune/"
echo ""

rsync -avz --progress -e "ssh -p $PORT" \
  ./dataset/ \
  $HOST:~/pdf-analyzer/finetune/dataset/

echo ""
echo "✅ Upload complete!"
