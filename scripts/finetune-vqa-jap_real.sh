#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=NONE
#SBATCH -w gnode76

bash prepare-vg-data.sh

module load cuda/10.0

EXPERIMENT_ID=$1

echo "Experiment ID: $EXPERIMENT_ID"
echo "Warming up the classifier for training..."

export DATASETS_DIR=/scratch/$USER/datasets
export MODELS_DIR=/scratch/$USER/models

export DATASET_DIR=$DATASETS_DIR/oscar/vg
export MODEL_DIR=$MODELS_DIR/vinvl-base/vg/$EXPERIMENT_ID

export NUM_LABELS=3000

export SSD=0

python3 ~/scripts/remove-classifier.py $MODEL_DIR/vqa/pytorch_model.bin

LAYERS="encoder2 encoder5 encoder8 encoder11"

echo "Found the following layers extracted on this node: $LAYERS"

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
    --num_train_epochs 5 \
    --output_dir $MODEL_DIR/classifier \
    --label_file $DATASET_DIR/trainval_ans2label.pkl \
    --save_epoch 1 \
    --seed 88 \
    --evaluate_during_training \
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
    --teacher_features_dir /ssd_scratch/cvit/$USER \
    --fp16 \
    --model_name_or_path $MODEL_DIR/vqa \
    --learning_rate 1e-04 \
    --lang ja \
    --only_classifier \
    # --do_kd \
    # --model_name_or_path /scratch/kshitij98/models/vinvl-base/vqa/checkpoint-2-14874 \
    # --learning_rate 1e-04 \
    # --model_name_or_path bert-base-multilingual-uncased \
    # --teacher_features_dir $DATASET_DIR/features/teacher \
    # --learning_rate 5e-05 \

echo "Training end-to-end..."

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
    --num_train_epochs 25 \
    --output_dir $MODEL_DIR/final \
    --label_file $DATASET_DIR/trainval_ans2label.pkl \
    --save_epoch 1 \
    --seed 88 \
    --evaluate_during_training \
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
    --teacher_features_dir /ssd_scratch/cvit/$USER \
    --fp16 \
    --model_name_or_path $MODEL_DIR/classifier/best \
    --learning_rate 5e-05 \
    --lang ja \
    # --do_kd \
    # --model_name_or_path /scratch/kshitij98/models/vinvl-base/vqa/checkpoint-2-14874 \
    # --learning_rate 1e-04 \
    # --model_name_or_path bert-base-multilingual-uncased \
    # --teacher_features_dir $DATASET_DIR/features/teacher \
    # --learning_rate 5e-05 \
