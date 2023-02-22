#!/bin/bash
#SBATCH -A nlp
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=NONE
#SBATCH -w gnode70

export NUM_LABELS=-1
export SSD=0

# Config

LANG=$1
MODEL=$2
EXPERIMENT_ID=$3

export DATA_DIR=/scratch/$USER/datasets/oscar/coco/$LANG
export FEATURES_DIR=/scratch/$USER/datasets/oscar/coco/features
export MODEL_DIR=/scratch/$USER/models/oscar/coco/$LANG

mkdir -p $MODEL_DIR/$EXPERIMENT_ID

echo "Training..."
echo "Language: $LANG"
echo "Model: $MODEL"
echo "EXPERIMENT_ID: $EXPERIMENT_ID"
echo "Outputs at $MODEL_DIR/$EXPERIMENT_ID"
echo "----------------------"

python oscar/run_retrieval_student.py \
    --lang $LANG \
    --model_name_or_path $MODEL \
    --do_train \
    --data_dir $DATA_DIR \
    --img_feat_file $FEATURES_DIR/features.tsv \
    --do_lower_case \
    --evaluate_during_training \
    --num_captions_per_img_train 1 \
    --num_captions_per_img_val 1 \
    --per_gpu_train_batch_size 16 \
    --learning_rate 0.00002 \
    --num_train_epochs 30 \
    --weight_decay 0.05 \
    --save_steps 1000 \
    --add_od_labels \
    --od_label_type vg \
    --max_seq_length 70 \
    --max_img_seq_length 70 \
    --output_dir $MODEL_DIR/$EXPERIMENT_ID \
    # --cross_image_eval \
    # --eval_caption_index_file minival_caption_indexs_top20.pt \
