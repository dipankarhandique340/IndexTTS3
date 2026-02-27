#!/bin/bash
# ==============================================================================
# IndexTTS2 CPU/Low-VRAM Setup Script
# Optimized for systems with <= 4GB VRAM or CPU-only
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$PATH"

echo "============================================="
echo "IndexTTS2 CPU/Low-VRAM Setup"
echo "============================================="

# Step 1: Create virtual environment with uv
echo ""
echo "[1/4] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv --python 3.10 .venv
    echo "  -> Virtual environment created."
else
    echo "  -> Virtual environment already exists."
fi

# Step 2: Install CPU-only PyTorch first (no CUDA dependency)
echo ""
echo "[2/4] Installing CPU-optimized PyTorch..."
uv pip install --python .venv/bin/python \
    torch==2.5.1+cpu \
    torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 3: Install remaining dependencies (skip flash-attn/triton/deepspeed)
echo ""
echo "[3/4] Installing project dependencies..."
uv pip install --python .venv/bin/python \
    accelerate>=1.2.0 \
    cn2an==0.5.22 \
    einops>=0.8.1 \
    ffmpeg-python==0.2.0 \
    g2p-en==2.1.0 \
    jieba==0.42.1 \
    json5==0.10.0 \
    librosa>=0.10.2 \
    matplotlib>=3.8.0 \
    modelscope>=1.20.0 \
    munch>=4.0.0 \
    numba>=0.58.0 \
    "numpy>=1.26.0,<2.0" \
    omegaconf>=2.3.0 \
    safetensors>=0.5.0 \
    sentencepiece>=0.2.1 \
    textstat>=0.7.10 \
    tokenizers>=0.21.0 \
    tqdm>=4.67.0 \
    transformers>=4.40.0 \
    huggingface-hub>=0.20.0 \
    soundfile>=0.12.0 \
    gradio>=5.0.0 \
    WeTextProcessing \
    pyyaml

# Step 4: Download model checkpoints
echo ""
echo "[4/4] Downloading IndexTTS2 model checkpoints..."
.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

print('Downloading IndexTTS-2 checkpoints from HuggingFace...')
print('This may take a while (~2GB download)...')
snapshot_download(
    repo_id='IndexTeam/IndexTTS-2',
    local_dir=checkpoint_dir,
    local_dir_use_symlinks=False
)
print('Checkpoints downloaded successfully!')
"

echo ""
echo "============================================="
echo "Setup complete!"
echo ""
echo "To run inference:"
echo "  .venv/bin/python infer_cpu.py"
echo ""
echo "To run test:"
echo "  .venv/bin/python test_cpu.py"
echo "============================================="
