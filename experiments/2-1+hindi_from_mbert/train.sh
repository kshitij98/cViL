#!/bin/bash

DATASET_DIR=/scratch/kshitij98/datasets
MODEL_DIR=/scratch/kshitij98/models

DATASET_DIR=$DATASET_DIR/oscar/vqa
MODEL_DIR=$MODEL_DIR/vinvl-base/vqa

python3 oscar/run_vqa.py -j 4 --img_feature_dim 2054 --max_img_seq_length
    50 --data_label_type mask --img_feature_type faster_r-cnn --data_dir $DATASET_DIR
    --model_type bert --model_name_or_path $MODEL_DIR/pytorch_model
    --task_name vqa_text --do_test --do_lower_case --max_seq_length 128 --per_gpu_eval_batch_size
    256 --per_gpu_train_batch_size 32 --learning_rate 5e-05 --num_train_epochs 1
    --output_dir $MODEL_DIR/results --label_file $DATASET_DIR/cache/trainval_ans2label.pkl
    --save_epoch 1 --seed 88 --evaluate_during_training --logging_steps 4000 --drop_out
    0.3 --weight_decay 0.05 --warmup_steps 0 --loss_type bce --img_feat_format pt 
    --classifier linear --cls_hidden_scale 3 --txt_data_dir $DATASET_DIR

