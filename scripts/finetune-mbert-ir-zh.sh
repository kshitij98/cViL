#!/bin/bash
#SBATCH -A nlp
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=NONE
#SBATCH -w gnode81

bash prepare-ir-data.sh

export NUM_LABELS=-1
export SSD=0

# Config

LANG=zh
MODEL=bert-base-multilingual-uncased
EXPERIMENT_ID=$1

export DATA_DIR=/scratch/$USER/datasets/oscar/coco/$LANG
export FEATURES_DIR=/scratch/$USER/datasets/oscar/coco/features
export MODEL_DIR=/scratch/$USER/models/oscar/coco/$LANG

mkdir -p $MODEL_DIR/$EXPERIMENT_ID
mkdir -p $MODEL_DIR/$EXPERIMENT_ID-classifier

echo "Training..."
echo "Language: $LANG"
echo "Model: $MODEL"
echo "EXPERIMENT_ID: $EXPERIMENT_ID"
echo "Outputs at $MODEL_DIR/$EXPERIMENT_ID"
echo "----------------------"

echo "Training classifier..."

python oscar/run_retrieval_student.py \
    --lang $LANG \
    --model_name_or_path $MODEL \
    --do_train \
    --data_dir $DATA_DIR \
    --img_feat_file $FEATURES_DIR/features.tsv \
    --do_lower_case \
    --per_gpu_train_batch_size 64 \
    --learning_rate 0.0001 \
    --num_train_epochs 5 \
    --weight_decay 0.05 \
    --save_steps 200 \
    --add_od_labels \
    --od_label_type vg \
    --max_seq_length 70 \
    --max_img_seq_length 70 \
    --output_dir $MODEL_DIR/$EXPERIMENT_ID-classifier \
    --only_classifier \
    --num_captions_per_img_val 40 \
    --eval_caption_index_file minival_caption_indexs_top40.pt \
    --evaluate_during_training \
    # --cross_image_eval \

# bash test-ir.sh ja $MODEL_DIR/$EXPERIMENT_ID/checkpoint-last $EXPERIMENT_ID

echo "Training end to end..."

python oscar/run_retrieval_student.py \
    --lang $LANG \
    --model_name_or_path $MODEL_DIR/$EXPERIMENT_ID-classifier/checkpoint-last \
    --do_train \
    --data_dir $DATA_DIR \
    --img_feat_file $FEATURES_DIR/features.tsv \
    --do_lower_case \
    --per_gpu_train_batch_size 16 \
    --learning_rate 0.00002 \
    --num_train_epochs 200 \
    --weight_decay 0.05 \
    --save_steps 5000 \
    --add_od_labels \
    --od_label_type vg \
    --max_seq_length 70 \
    --max_img_seq_length 70 \
    --output_dir $MODEL_DIR/$EXPERIMENT_ID \
    --num_workers 8 \
    --evaluate_during_training \
    --eval_caption_index_file minival_caption_indexs_top100.pt \
    --num_captions_per_img_val 100 \

    # --only_classifier \
    # --cross_image_eval \

bash test-ir.sh $LANG $MODEL_DIR/$EXPERIMENT_ID/checkpoint-last $EXPERIMENT_ID
