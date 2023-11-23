#!/usr/bin/env bash
set -e          # 只要脚本发生错误就停止运行
set -u          # 如果遇到不存在的变量就报错并停止执行
set -x          # 运行指令结果的时候，输出对应的指令
set -o pipefail # 确保只要一个子命令失败，整个管道命令就失败

export CUDA_VISIBLE_DEVICES="$1"
echo 'Excute the script on GPU: ' "$1"

echo 'For COD'
python test.py --config ./configs/zoomnet/zoomnet.py \
    --model-name ZoomNet \
    --batch-size 22 \
    --load-from /home/suyoung/PycharmProjects/COD-Project/Ours/output/ZoomNet_BS16_LR0.01_E100_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/pth/state_final.pth \
    --save-path ./output/ForSharing/COD_Results_1

#echo 'For SOD'
#python test.py --config ./configs/ento/sod_zoomnet.py \
#    --model-name ZoomNet \
#    --batch-size 22 \
#    --load-from ./output/ForSharing/sod_zoomnet_r50_bs22_e50_2022-03-04_fixed.pth \
#    --save-path ./output/ForSharing/SOD_Results
