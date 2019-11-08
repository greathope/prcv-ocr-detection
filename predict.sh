#!/usr/bin/env

CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_test.sh configs/ocr/cascade_dcn.py \
    /gruntdata/openImages/ocr/models/cascade_rcnn_dcn_x101_32x4d_fpn_1x/latest.pth 8 \
    --out /gruntdata/openImages/ocr/models/cascade_rcnn_dcn_x101_32x4d_fpn_1x/results_train.pkl --eval bbox

