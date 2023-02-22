#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode85

source prepare-vg-data.sh

MODEL_ID=$1

export DATASETS_DIR=/scratch/$USER/datasets
export MODELS_DIR=/scratch/$USER/models

export DATASET_DIR=$DATASETS_DIR/oscar/vg
export MODEL_DIR=$MODELS_DIR/vinvl-base/vg/$MODEL_ID

export SSD=1
export TEACHER_DIR=$DATASET_DIR
[[ "$SSD" == "1" ]] && export TEACHER_DIR=/ssd_scratch/cvit/$USER
[[ "$SSD" == "1" ]] && [[ "$USER" == "devanshg27" ]] && export TEACHER_DIR=/ssd_scratch/users/devanshg27

# export TEACHER_DIR=/ssd_scratch/cvit/$USER

export NUM_LABELS=3000

mkdir $TEACHER_DIR/$MODEL_ID -p

LAYERS="encoder10 encoder11"

echo "Extracting the following layers: $LAYERS"

python3 oscar/extract_vqa_features.py -j 4 \
    --img_feature_dim 2054 \
    --max_img_seq_length 50 \
    --data_label_type mask \
    --img_feature_type faster_r-cnn \
    --data_dir $DATASET_DIR \
    --model_type bert \
    --model_name_or_path $MODEL_DIR \
    --task_name vqa_text \
    --do_eval \
    --do_lower_case \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size 128 \
    --per_gpu_train_batch_size 16 \
    --learning_rate 5e-05 \
    --num_train_epochs 1 \
    --output_dir $MODEL_DIR \
    --label_file $DATASET_DIR/trainval_ans2label.pkl \
    --save_epoch 1 \
    --seed 88 \
    --evaluate_during_training \
    --logging_steps 200 \
    --drop_out 0.3 \
    --weight_decay 0.05 \
    --warmup_steps 0 \
    --loss_type bce \
    --img_feat_format pt  \
    --classifier linear \
    --cls_hidden_scale 3 \
    --txt_data_dir $DATASET_DIR \
    --layers $LAYERS \
    --teacher_features_dir $TEACHER_DIR/$MODEL_ID \
    --lang ja

if [ ! -f "$TEACHER_DIR/layers.txt" ]; then
    for LAYER in $LAYERS; do
        echo $LAYER >> $TEACHER_DIR/layers.txt;
    done
fi
