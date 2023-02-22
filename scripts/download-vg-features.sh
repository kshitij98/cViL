#!/bin/bash

mkdir -p /scratch/$USER/datasets/oscar/vg

azcopy copy https://biglmdiag.blob.core.windows.net/vinvl/image_features/gqa_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/ /scratch/$USER/datasets/oscar/vg/features --recursive
