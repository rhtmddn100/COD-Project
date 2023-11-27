# ENTO: Decoders for Decoder in Camouflaged Object Detection



## Usage

### Dependencies

Some core dependencies:

- timm == 0.4.12
- torch == 1.8.1
- [pysodmetrics](https://github.com/lartpang/PySODMetrics) == 1.2.4 # for evaluating results

More details can be found in <./requirements.txt>

### Datasets

COD Datasets can be downloaded at:
- COD Datasets: <https://github.com/lartpang/awesome-segmentation-saliency-dataset#camouflaged-object-detection-cod>

Put the COD dataset into TrainDataset and TestDataset. 

Refer to configs/base/dataset/dataset_configs.json for the dataset directory.

### Download backbone
The pretrained model is stored in [Google Drive](https://drive.google.com/file/d/1H3UeZzOk7KL7_-SkUvk6Qijjq_dQrE98/view?usp=share_link). After downloading, please put it in the pretrained_pvt folder.

Backbone model paper is [PVT v2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)


### Training

You can use our default configuration like this:

```shell
$ python main.py --model-name=ENTO --config=configs/ento/ento.py --datasets-info ./configs/_base_/dataset/dataset_configs.json --info demo
```
