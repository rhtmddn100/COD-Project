#!/bin/sh
#SBATCH -J suyoung
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 2

#SBATCH -o suyoung_resnet.out
#SBATCH -e suyoung_resnet.err
#SBATCH --time 12:00:00
#SBATCH --gres=gpu:1

python main.py --model-name=ZoomNet --config=configs/zoomnet/zoomnet.py --datasets-info ./configs/_base_/dataset/dataset_configs.json --info demo > suyoung_resnet.out
