#!/bin/bash
#SBATCH -A research
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode71

EXPERIMENT_ID=$1

bash train-ir.sh zh bert-base-multilingual-uncased $EXPERIMENT_ID