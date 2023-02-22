#!/bin/bash

DATASET_DIR=/scratch/$USER/datasets/oscar
mkdir -p $DATASET_DIR

echo "CAUTION: Check job on gnode85"

rsync -rah samyakxd@gnode85:/scratch/kshitij98/datasets/oscar/vg $DATASET_DIR --ignore-existing --info=progress2
rsync -rah samyakxd@gnode85:/scratch/kshitij98/datasets/oscar/vg/*.json $DATASET_DIR --info=progress2
rsync -rah samyakxd@gnode85:/scratch/kshitij98/datasets/oscar/vg/*.pkl $DATASET_DIR --info=progress2

echo "Data ready!"

