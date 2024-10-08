#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox

# /media/max/GAME/MA/SGN/tools/dist_test.sh /media/max/GAME/MA/SGN/projects/configs/sgn/sgn-T-one-stage-guidance.py /media/max/GAME/MA/SGN/ckpt/sgn-t-epoch_25.pth 1
