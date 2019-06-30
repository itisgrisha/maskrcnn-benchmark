#!/bin/bash

exp={1:-ice}

git pull

mkdir -p /data/mask_ckpts/$exp

python3 tools/train_net.py \
  --config-file=configs/efnet_retina.yaml \
  OUTPUT_DIR /data/mask_ckpts/$exp |& tee /data/mask_ckpts/$exp/log.txt
