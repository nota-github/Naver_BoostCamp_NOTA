#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0

GPUS=$1
PORT=${PORT:-9998}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env main.py 
