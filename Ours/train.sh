#!/bin/sh
#SBATCH -J suyoung
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 2

#SBATCH -o suyoung01.out
#SBATCH -e suyoung01.err
#SBATCH --time 12:00:00
#SBATCH --gres=gpu:1

python main.py --model-name=ZoomNet --config=configs/zoomnet/zoomnet.py --datasets-info ./configs/_base_/dataset/dataset_configs.json --info demo > suyoung01.out
