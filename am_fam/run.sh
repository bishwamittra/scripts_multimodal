#!/bin/bash

CUDA_VISIBLE_DEVICE=0 python main.py \
    --seed=42 \
    --epoch=100 \
    --batch_size=12 \
    --lr=1e-4



