#!/bin/bash
#SBATCH -A research
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=NONE
#SBATCH -w gnode32

LANG=$1

bash prepare-ir-data.sh

MODELS_DIR=/scratch/$USER/models/coco/$LANG
mkdir -p $MODELS_DIR

rsync -rah devanshg27@ada:/share1/devanshg27/models/vinvl/coco/$LANG/baseline $MODELS_DIR --ignore-existing --info=progress2

bash test-ir.sh $LANG $MODELS_DIR/baseline
