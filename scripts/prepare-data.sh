#!/bin/bash

KSHITIJ_DATASET=/scratch/kshitij98/datasets/oscar
DATASET=/scratch/$USER/datasets/oscar
MODEL=/scratch/$USER/models/vinvl-base
KSHITIJ_MODEL=/scratch/kshitij98/models/vinvl-base

mkdir -p $DATASET
mkdir -p $MODEL

NODE=0
# NODE=gnode85
[[ $# -gt 0 ]] && NODE=$1

echo $NODE

if [ ! -d "$DATASET/vqa" ]; then
    [[ $NODE == 0 ]] && azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/datasets/vqa' $DATASET --recursive

    [[ $NODE != 0 ]] && mkdir $DATASET/vqa
fi
[[ $NODE != 0 ]] && rsync -rah samyakxd@$NODE:$KSHITIJ_DATASET/vqa/*.pt $DATASET/vqa --info=progress2 --ignore-existing
[[ $NODE != 0 ]] && rsync -rah samyakxd@$NODE:$KSHITIJ_DATASET/vqa/*.json $DATASET/vqa --info=progress2 --ignore-existing
[[ $NODE != 0 ]] && rsync -rah samyakxd@$NODE:$KSHITIJ_DATASET/vqa/*.pkl $DATASET/vqa --info=progress2 --ignore-existing
[[ $NODE != 0 ]] && rsync -rah samyakxd@$NODE:$KSHITIJ_DATASET/vqa/*.tsv $DATASET/vqa --info=progress2 --ignore-existing

if [ ! -d "$MODEL/vqa" ]; then
    rsync -rah kshitij98@ada:/share3/kshitij98/models/oscar/vinvl_base_vqa/best.zip /scratch/$USER/models --info=progress2 --ignore-existing
    unzip /scratch/$USER/models/best.zip -d $MODEL
    mv $MODEL/best/best $MODEL/vqa
    rmdir $MODEL/best
fi

mkdir -p $MODEL/pretrained
[[ $NODE != 0 ]] && rsync -rah samyakxd@$NODE:$KSHITIJ_MODEL/pretrained/* $MODEL/pretrained --info=progress2 --ignore-existing

if [ ! -d "$DATASET/vqa/train" ]; then
    LASTDIR=$PWD
    cp features-split.py $DATASET/vqa/
    cd $DATASET/vqa/
    python3 features-split.py
    cd $LASTDIR
fi

echo "Data ready!"
