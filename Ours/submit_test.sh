#!/bin/bash

#SBATCH --job-name=cod_eval       # Submit a job named "example"
#SBATCH --partition=a3000        # 계산노드 종류 선택: a6000 or a100
#SBATCH --gres=gpu:1            # Use 1 GPU
#SBATCH --time=12-24:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=32000              # cpu memory size
#SBATCH --cpus-per-task=8        # cpu 개수
#SBATCH --output=log__.txt         # 스크립트 실행 결과 std output을 저장할 파일 이름

ml purge
ml load cuda/11.1                # 필요한 쿠다 버전 로드
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate project_cod             # Activate your conda environment

srun --unbuffered python main.py --model-name=ZoomNet --config=configs/zoomnet/zoomnet.py --datasets-info ./configs/_base_/dataset/dataset_configs.json --info demo