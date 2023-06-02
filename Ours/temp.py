import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
import os
import sys

# from methods.module.base_model import BasicModelClass
# from methods.module.conv_block import ConvBNReLU
# from utils.builder import MODELS
# from utils.ops import cus_sample
# from methods.module.pvtv2 import pvt_v2_b2
# from utils import builder, configurator, io, misc, ops, pipeline, recorder

# model, model_code = builder.build_obj_from_registry(
#         registry_name="MODELS", obj_name="ZoomNet", return_code=True
#     )
#
# param_groups = {}
# for name, param in model.named_parameters():
#     # print(name)
#     if name.startswith("backbone.block"):
#         param_groups.setdefault("pretrained", []).append(param)
#     elif name.startswith("backbone"):
#         param_groups.setdefault("fixed", []).append(param)
#     else:
#         param_groups.setdefault("retrained", []).append(param)
#
#
# print(param_groups["pretrained"])


# Load the model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

input_tensor = torch.randn(8, 3, 384, 384)  # 임의의 입력 텐서
output_tensor = model.get_intermediate_layers(input_tensor, n=4)
for t in output_tensor:
    print(t.shape)
