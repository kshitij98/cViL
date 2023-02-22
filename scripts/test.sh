#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode59

bash prepare-data.sh

module load cuda/10.0

export DATASETS_DIR=/scratch/$USER/datasets
export MODELS_DIR=/scratch/$USER/models

export DATASET_DIR=$DATASETS_DIR/oscar/vqa
export MODEL_DIR=$MODELS_DIR/vinvl-base/vqa

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
    --max_seq_length 128 \
    --per_gpu_eval_batch_size 128 \
    --per_gpu_train_batch_size 32 \
    --num_train_epochs 1 \
    --output_dir $MODEL_DIR \
    --label_file $DATASET_DIR/trainval_ans2label.pkl \
    --label2ans_file $DATASET_DIR/trainval_label2ans.pkl \
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
    --model_name_or_path $MODEL_DIR \
    --learning_rate 1e-04 \
    --lang en \
    --do_eval \
    --do_test_dev \
    # --do_train \
    # --do_kd \
    # --fp16 \
    # --model_name_or_path /scratch/kshitij98/models/vinvl-base/vqa/checkpoint-2-14874 \
    # --learning_rate 1e-04 \
    # --model_name_or_path bert-base-multilingual-uncased \
    # --teacher_features_dir $DATASET_DIR/features/teacher \
    # --learning_rate 5e-05 \



# #!/bin/bash

# DATASET_DIR=/scratch/kshitij98/datasets
# MODEL_DIR=/scratch/kshitij98/models

# DATASET_DIR=$DATASET_DIR/oscar/vqa
# MODEL_DIR=$MODEL_DIR/vinvl-base/vqa/

# python3 oscar/run_vqa.py -j 4 \
#     --img_feature_dim 2054 \
#     --max_img_seq_length 50 \
#     --data_label_type mask \
#     --img_feature_type faster_r-cnn \
#     --data_dir $DATASET_DIR \
#     --model_type bert \
#     --model_name_or_path $MODEL_DIR \
#     --task_name vqa_text \
#     --do_eval \
#     --max_seq_length 128 \
#     --per_gpu_eval_batch_size 128 \
#     --per_gpu_train_batch_size 32 \
#     --learning_rate 5e-05 \
#     --num_train_epochs 1 \
#     --output_dir $MODEL_DIR \
#     --label_file $DATASET_DIR/trainval_ans2label.pkl \
#     --save_epoch 1 \
#     --seed 88 \
#     --evaluate_during_training \
#     --logging_steps 4000 \
#     --drop_out 0.3 \
#     --weight_decay 0.05 \
#     --warmup_steps 0 \
#     --loss_type bce \
#     --img_feat_format pt  \
#     --classifier linear \
#     --cls_hidden_scale 3 \
#     --txt_data_dir $DATASET_DIR \
#     --do_lower_case \
#     --do_test_dev \
