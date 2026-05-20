#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/fatty_liver_project

/root/miniconda3/bin/python run_experiments.py \
  --models resnet18 mobilenet_v3_small efficientnet_b0 \
  --seeds 42 43 44 \
  --epochs 30 \
  --batch-size 32 \
  --loss weighted_ce
