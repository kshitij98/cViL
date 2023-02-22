export DATA_DIR=/scratch/$USER/datasets/oscar/coco/en
export FEATURES_DIR=/scratch/$USER/datasets/oscar/coco/features
export MODEL_DIR=/scratch/$USER/models/oscar/coco/en
export NUM_LABELS=-1
export SSD=0

LANG=$1
MODEL=$2

echo "Testing..."
echo "Language: $LANG"
echo "Model: $MODEL"
echo "----------------------"

python oscar/run_retrieval_student.py \
    --do_eval \
    --num_captions_per_img_val 5 \
    --per_gpu_eval_batch_size 64 \
    --data_dir $DATA_DIR \
    --eval_model_dir $MODEL \
    --lang $LANG \
    --img_feat_file $FEATURES_DIR/features.tsv \
    --test_split minival \
    --eval_img_keys_file $DATA_DIR/minival_img_keys.tsv \
    --eval_caption_index_file minival_caption_indexs_top20.pt \
