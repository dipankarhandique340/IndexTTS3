#!/bin/bash

# Make sure you are in the root directory of the project when running this
# Usage: ./training_lightweight_en/train_lightweight_s2mel.sh

echo "Starting S2Mel lightweight TURBO training..."
python trainers/train_s2mel_v2.py \
    --train-manifest datasets/en_processed_data/train_manifest.jsonl \
    --val-manifest datasets/en_processed_data/val_manifest.jsonl \
    --config training_lightweight_en/config_light.yaml \
    --base-checkpoint training_lightweight_en/s2mel_light_init.pth \
    --output-dir checkpoints_lightweight_s2mel \
    --audio-root datasets/LJSpeech-1.1/wavs \
    --resume auto \
    --batch-size 32 \
    --grad-accumulation 2 \
    --epochs 30 \
    --learning-rate 5e-4 \
    --val-interval 1000 \
    --amp
