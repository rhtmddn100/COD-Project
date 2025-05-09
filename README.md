# A Revisit to The Decoder for the Camouflage Object Detection

[![BMVC 2024 Acceptance](https://img.shields.io/badge/Conference-BMVC%202024-blue)](https://bmvc2024.org/) [![arXiv](https://img.shields.io/badge/arXiv-2309.14495-red)](https://arxiv.org/abs/2309.14495) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![Torch](https://img.shields.io/badge/torch-1.8.1-orange)](https://pytorch.org/) 

## Introduction

This repository is the **official implementation** of the paper [“A Revisit to the Decoder for Camouflage Object Detection”](https://arxiv.org/abs/2309.14495), accepted at **BMVC 2024**. We revisit decoder designs for Camouflaged Object Detection (COD) and introduce a novel architecture achieving state-of-the-art performance on standard benchmarks.

## Architecture
![Architecture](./architecture.png)

## 🛠️ Dependencies

Some core dependencies:

- timm == 0.4.12
- torch == 1.8.1
- [pysodmetrics](https://github.com/lartpang/PySODMetrics) == 1.2.4 # for evaluating results

More details can be found in _requirements.txt_

## 📂 Datasets

COD Datasets can be downloaded at:
- COD Datasets: <https://github.com/lartpang/awesome-segmentation-saliency-dataset#camouflaged-object-detection-cod>

Put the COD dataset into TrainDataset and TestDataset. \
Refer to configs/base/dataset/dataset_configs.json for the dataset directory.

## ⚙️ Pretrained backbone
The pretrained model is stored in [Google Drive](https://drive.google.com/file/d/1H3UeZzOk7KL7_-SkUvk6Qijjq_dQrE98/view?usp=share_link). After downloading, please put it in the pretrained_pvt folder.

_Backbone paper_: [PVT v2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)



## Training

You can use our default configuration like this:

```shell
python main.py \
  --model-name=ENTO \
  --config=configs/ento/ento.py \
  --datasets-info=./configs/_base_/dataset/dataset_configs.json \
  --info=demo