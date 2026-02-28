#!/bin/bash

set -e

WEIGHTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/weights"
HF_BASE="https://huggingface.co/FF2416/AC-Foley/resolve/main/weights"

mkdir -p "$WEIGHTS_DIR"

echo "Downloading weights to: $WEIGHTS_DIR"

FILES=(
    "ac_foley.pth"
    "best_netG.pt"
    "empty_string.pth"
    "synchformer_state_dict.pth"
    "v1-44.pth"
)

for FILE in "${FILES[@]}"; do
    TARGET="$WEIGHTS_DIR/$FILE"
    if [ -f "$TARGET" ]; then
        echo "[skip] $FILE already exists."
    else
        echo "[download] $FILE ..."
        wget -q --show-progress -O "$TARGET" "$HF_BASE/$FILE"
    fi
done

echo "All weights downloaded to $WEIGHTS_DIR"