#!/usr/bin/env bash
set -e
set -u
set -x
set -o pipefail

export CUDA_VISIBLE_DEVICES="$1"
echo 'Excute the script on GPU: ' "$1"

echo 'For COD'
python test.py --config ./configs/ento/ento.py \
    --model-name ETNO \
    --batch-size 22 \
    --load-from /home/suyoung/PycharmProjects/COD-Project/Ours/output/ENTO_BS16_LR0.01_E100_H768_W768_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/pth/state_final.pth \
    --save-path ./output/ForSharing/COD_Results_1

#echo 'For SOD'
#python test.py --config ./configs/ento/sod_zoomnet.py \
#    --model-name ZoomNet \
#    --batch-size 22 \
#    --load-from ./output/ForSharing/sod_zoomnet_r50_bs22_e50_2022-03-04_fixed.pth \
#    --save-path ./output/ForSharing/SOD_Results
