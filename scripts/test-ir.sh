#!/bin/bash
#SBATCH -A research
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=NONE
#SBATCH -w gnode32

bash prepare-ir-data.sh

export NUM_LABELS=-1
export SSD=0

# Config

LANG=$1
MODEL=$2
TEST_FILE=test_img_keys_1k.tsv
[[ $LANG == zh ]] && TEST_FILE=test_img_keys.tsv
# EXPERIMENT_ID=$3

export DATA_DIR=/scratch/$USER/datasets/oscar/coco/$LANG
export FEATURES_DIR=/scratch/$USER/datasets/oscar/coco/features
export MODEL_DIR=/scratch/$USER/models/oscar/coco/$LANG

# mkdir -p $MODEL_DIR/$EXPERIMENT_ID

echo "Testing..."
echo "Language: $LANG"
echo "Model: $MODEL"
# echo "EXPERIMENT_ID: $EXPERIMENT_ID"
echo "Added vg labels with the input!!!"
echo "----------------------"

python oscar/run_retrieval_student.py \
    --lang $LANG \
    --do_eval \
    --do_test \
    --cross_image_eval \
    --per_gpu_eval_batch_size 64 \
    --data_dir $DATA_DIR \
    --eval_model_dir $MODEL \
    --img_feat_file $FEATURES_DIR/features.tsv \
    --test_split test \
    --eval_img_keys_file $DATA_DIR/$TEST_FILE \
    --add_od_labels \
    --od_label_type vg \
    # --num_captions_per_img_val 5 \
    # --max_img_seq_length 70 \
    # --eval_caption_index_file minival_caption_indexs_top20.pt \
    # --test_split test \
    # --eval_img_keys_file $DATA_DIR/test_img_keys_1k.tsv \

