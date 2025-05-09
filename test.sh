#!/usr/bin/env bash
set -e
set -u
set -x
set -o pipefail

export CUDA_VISIBLE_DEVICES="$1"
echo 'Excute the script on GPU: ' "$1"

echo 'For COD'
python test.py --config ./configs/ento/ento.py \
    --model-name ENTO \
    --batch-size 22 \
    --load-from /home/suyoung/PycharmProjects/COD-Project/Ours/output/ENTO_BS16_LR0.01_E100_H768_W768_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/pth/state_final.pth \
    --save-path ./output/ForSharing/COD_Results_1
