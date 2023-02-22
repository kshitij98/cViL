#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode89

bash prepare-vg-data.sh

module load cuda/10.0

EXPERIMENT_ID=$1

echo "CAUTION: Check models in Experiment directory"
echo "Experiment ID: $EXPERIMENT_ID"

export DATASETS_DIR=/scratch/$USER/datasets
export MODELS_DIR=/scratch/$USER/models

export DATASET_DIR=$DATASETS_DIR/oscar/vg
export DATASET_DIR_COPY=$DATASET_DIR.test
export MODEL_DIR=$MODELS_DIR/vinvl-base/vg/$EXPERIMENT_ID

mkdir -p $DATASET_DIR_COPY
for file in $(ls $DATASET_DIR); do
    ln -s $DATASET_DIR/$file $DATASET_DIR_COPY/$file 
done

rm $DATASET_DIR_COPY/val
rm $DATASET_DIR_COPY/val2014_qla_mrcnn.json

mv $DATASET_DIR_COPY/test-dev2015 $DATASET_DIR_COPY/val
mv $DATASET_DIR_COPY/test-dev2015_qla_mrcnn.json $DATASET_DIR_COPY/val2014_qla_mrcnn.json

export NUM_LABELS=3000

export SSD=0

LAYERS="encoder2 encoder5 encoder8 encoder11"

python3 oscar/run_vqa_student.py -j 8 \
    --img_feature_dim 2054 \
    --max_img_seq_length 50 \
    --data_label_type mask \
    --img_feature_type faster_r-cnn \
    --data_dir $DATASET_DIR_COPY \
    --model_type bert \
    --task_name vqa_text \
    --do_eval \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size 128 \
    --per_gpu_train_batch_size 32 \
    --num_train_epochs 15 \
    --output_dir $MODEL_DIR/final \
    --label_file $DATASET_DIR_COPY/trainval_ans2label.pkl \
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
    --txt_data_dir $DATASET_DIR_COPY \
    --layers $LAYERS \
    --do_lower_case \
    --gradient_accumulation_steps 1 \
    --teacher_features_dir /ssd_scratch/cvit/$USER \
    --fp16 \
    --model_name_or_path $MODEL_DIR/final/best \
    --learning_rate 5e-05 \
    --lang ja \
    # --do_kd \
    # --model_name_or_path /scratch/kshitij98/models/vinvl-base/vqa/checkpoint-2-14874 \
    # --learning_rate 1e-04 \
    # --model_name_or_path bert-base-multilingual-uncased \
    # --teacher_features_dir $DATASET_DIR/features/teacher \
    # --learning_rate 5e-05 \


echo "Multiply score with the following:"
echo "224185 / 301392"