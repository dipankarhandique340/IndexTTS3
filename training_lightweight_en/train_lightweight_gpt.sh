#!/bin/bash

# Make sure you are in the root directory of the project when running this
# Usage: ./training_lightweight_en/train_lightweight.sh

echo "Starting GPT lightweight training..."
uv run python trainers/train_gpt_v2.py \
    --train-manifest path/to/your/english_train_data.jsonl \
    --val-manifest path/to/your/english_val_data.jsonl \
    --tokenizer checkpoints/bpe.model \
    --config training_lightweight_en/config_light.yaml \
    --base-checkpoint training_lightweight_en/gpt_light_init.pth \
    --output-dir checkpoints_lightweight_gpt \
    --batch-size 32 \
    --grad-accumulation 1 \
    --epochs 50 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --warmup-steps 1000 \
    --log-interval 1 \
    --val-interval 2000 \
    --grad-clip 1.0 \
    --text-loss-weight 0.2 \
    --mel-loss-weight 0.8 \
    --amp
