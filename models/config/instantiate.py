# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

from detectron2.config import instantiate

def instantiate_odise(cfg):
    """
    实例化 ODISE（可能是一个目标检测或图像分割模型）模型。

    此函数的主要功能是根据传入的配置对象 `cfg` 实例化模型的骨干网络（backbone），
    并更新语义分割头（sem_seg_head）及其像素解码器（pixel_decoder）的输入形状，
    最后根据更新后的配置实例化整个模型。

    参数:
        cfg (Config): 包含模型配置信息的对象，通常是 Detectron2 风格的配置对象。
                      该对象应包含 `backbone`、`sem_seg_head` 等相关配置项。

    返回:
        Model: 实例化后的 ODISE 模型。
    """
    backbone = instantiate(cfg.backbone)
    cfg.sem_seg_head.input_shape = backbone.output_shape()
    cfg.sem_seg_head.pixel_decoder.input_shape = backbone.output_shape()
    cfg.backbone = backbone
    model = instantiate(cfg)

    return model