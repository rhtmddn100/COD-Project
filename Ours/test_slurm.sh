#!/bin/sh
#SBATCH -J suyoung
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 2

#SBATCH -o suyoung01.out
#SBATCH -e suyoung01.err
#SBATCH --time 12:00:00
#SBATCH --gres=gpu:2

python test.py --config ./configs/zoomnet/zoomnet.py \
    --model-name ZoomNet \
    --batch-size 22 \
    --load-from /home/user/suyoung/COD-Project/Ours/output/ZoomNet_BS16_LR0.01_E100_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/pth/state_final.pth \
    --save-path ./output/ForSharing/COD_Results_1 > suyoung_test1.out
