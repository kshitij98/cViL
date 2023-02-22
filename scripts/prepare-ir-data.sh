#!/bin/bash

DATASET_DIR=/scratch/$USER/datasets/oscar/coco
MODEL_DIR=/scratch/$USER/models/oscar/coco
mkdir -p $DATASET_DIR
mkdir -p $MODEL_DIR

echo "CAUTION: Check job on gnode71"

rsync -rah samyakxd@gnode71:/scratch/kshitij98/datasets/oscar/coco/en $DATASET_DIR --ignore-existing --info=progress2
rsync -rah samyakxd@gnode71:/scratch/kshitij98/datasets/oscar/coco/ja $DATASET_DIR --ignore-existing --info=progress2
rsync -rah samyakxd@gnode71:/scratch/kshitij98/datasets/oscar/coco/zh $DATASET_DIR --ignore-existing --info=progress2
rsync -rah samyakxd@gnode71:/scratch/kshitij98/datasets/oscar/coco/zh-aug $DATASET_DIR --ignore-existing --info=progress2
rsync -rah samyakxd@gnode71:/scratch/kshitij98/datasets/oscar/coco/features $DATASET_DIR --ignore-existing --info=progress2
rsync -rah samyakxd@gnode71:/scratch/kshitij98/models/oscar/coco/en $MODEL_DIR --ignore-existing --info=progress2

echo "Data ready!"
