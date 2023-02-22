#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode84

module load cuda/10.0

EXPERIMENT_ID=$1

bash extract-features.sh

# bash train-student.sh $EXPERIMENT_ID 1
bash train-student-kd.sh $EXPERIMENT_ID 1