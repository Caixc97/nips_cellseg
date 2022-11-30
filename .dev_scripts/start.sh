#!/usr/bin/env bash

GPU_ID=$1

sudo docker run --rm -it --gpus '"device='all'"' --ipc=host \
    -v $PWD/sartorius:/workspace \
    -v $PWD/kaggle_data:/data \
    root/redcat-cell-instance-segmentation
