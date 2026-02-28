#!/bin/bash

# Make sure you are in the root directory of the project when running this
# Usage: ./training_lightweight_en/train_lightweight_s2mel.sh

echo "Starting S2Mel lightweight training..."
python trainers/train_s2mel_v2.py \
    --train-manifest path/to/your/english_train_data.jsonl \
    --val-manifest path/to/your/english_val_data.jsonl \
    --config training_lightweight_en/config_light.yaml \
    --base-checkpoint training_lightweight_en/s2mel_light_init.pth \
    --output-dir checkpoints_lightweight_s2mel \
    --batch-size 16 \
    --grad-accumulation 2 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --val-interval 2000 \
    --amp
