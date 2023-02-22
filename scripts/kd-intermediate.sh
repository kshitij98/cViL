#!/bin/bash
#SBATCH -A nlp
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode82

bash extract-features.sh
bash prepare-data.sh

module load cuda/10.0

EXPERIMENT_ID=$1

echo "Training entire model...!!! for $EXPERIMENT_ID"

echo "Experiment ID: $EXPERIMENT_ID"

export DATASETS_DIR=/scratch/$USER/datasets
export MODELS_DIR=/scratch/$USER/models

export DATASET_DIR=$DATASETS_DIR/oscar/vqa
export MODEL_DIR=$MODELS_DIR/vinvl-base/vqa/$EXPERIMENT_ID

export NUM_LABELS=3129
export SSD=1
export FINAL_LAYER_ONLY=1

FEATURE_DIR=$DATASET_DIR

[[ "$SSD" == "1" ]] && FEATURE_DIR=/ssd_scratch/cvit/$USER
[[ "$SSD" == "1" ]] && [[ "$USER" == "devanshg27" ]] && FEATURE_DIR=/ssd_scratch/users/devanshg27

# Currently disabled! Manually set in code!
LAYERS="encoder11"

# echo "Found the following layers extracted on this node: $LAYERS"

python3 oscar/run_vqa_student.py -j 8 \
    --img_feature_dim 2054 \
    --max_img_seq_length 50 \
    --data_label_type mask \
    --img_feature_type faster_r-cnn \
    --data_dir $DATASET_DIR \
    --model_type bert \
    --task_name vqa_text \
    --do_train \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size 128 \
    --per_gpu_train_batch_size 32 \
    --num_train_epochs 10 \
    --output_dir $MODEL_DIR/kd-intermediate \
    --label_file $DATASET_DIR/trainval_ans2label.pkl \
    --save_epoch 1 \
    --seed 88 \
    --logging_steps 500 \
    --drop_out 0.3 \
    --weight_decay 0.05 \
    --warmup_steps 0 \
    --loss_type bce \
    --img_feat_format pt  \
    --classifier linear \
    --cls_hidden_scale 3 \
    --txt_data_dir $DATASET_DIR \
    --layers $LAYERS \
    --do_lower_case \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --model_name_or_path bert-base-multilingual-uncased \
    --learning_rate 5e-05 \
    --lang ja \
    --do_kd \
    --teacher_features_dir $FEATURE_DIR \
    --codemix \
    # --no_tags \
    # --no_images \
    # --teacher_features_dir /ssd_scratch/cvit/$USER \
    # --model_name_or_path /scratch/kshitij98/models/vinvl-base/vqa/checkpoint-2-14874 \
    # --learning_rate 1e-04 \
    # --model_name_or_path bert-base-multilingual-uncased \
    # --teacher_features_dir $DATASET_DIR/features/teacher \
    # --learning_rate 5e-05 \
