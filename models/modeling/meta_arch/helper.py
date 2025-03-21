# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm for channels of '2D' spatial NCHW tensors.
    This class inherits from nn.LayerNorm and is designed to apply layer normalization
    on the channel dimension of 2D spatial tensors in NCHW format.
    """

    def __init__(self, num_channels, eps=1e-6, affine=True):
        """
        初始化LayerNorm2d类的实例。

        参数:
            num_channels (int): 输入张量的通道数。
            eps (float, optional): 为了数值稳定性添加到分母的一个小常数，默认为1e-6。
            affine (bool, optional): 是否使用可学习的仿射变换参数（权重和偏置），默认为True。
        """
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，对输入张量进行层归一化操作。

        参数:
            x (torch.Tensor): 输入的2D空间张量，形状为 [N, C, H, W]。

        返回:
            torch.Tensor: 经过层归一化后的张量，形状与输入相同。
        """
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)


class FeatureExtractor(nn.Module, metaclass=ABCMeta):
    """
    FeatureExtractor 是一个抽象基类，用于定义特征提取器的基本接口。
    它继承自 nn.Module，并使用 ABCMeta 元类来定义抽象方法。
    """

    def __init__(self):
        """
        初始化 FeatureExtractor 类的实例。
        """
        super().__init__()

    def ignored_state_dict(self, destination=None, prefix=""):
        """
        获取模型的状态字典，忽略某些模块的状态。

        参数:
            destination (OrderedDict, optional): 用于存储状态字典的目标字典，默认为 None。
            prefix (str, optional): 状态字典中键的前缀，默认为空字符串。

        返回:
            OrderedDict: 包含模型状态的字典。
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self._modules.items():
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return super().state_dict(destination=destination, prefix=prefix)

    # don't save DDPM model
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        重写 state_dict 方法，返回一个空的 OrderedDict，不保存模型状态。

        参数:
            destination (OrderedDict, optional): 用于存储状态字典的目标字典，默认为 None。
            prefix (str, optional): 状态字典中键的前缀，默认为空字符串。
            keep_vars (bool, optional): 是否保留变量，默认为 False。

        返回:
            OrderedDict: 一个空的 OrderedDict。
        """
        return OrderedDict()

    def train(self, mode: bool = True):
        """
        设置模型的训练模式，并冻结模型参数。

        参数:
            mode (bool, optional): 是否为训练模式，默认为 True。

        返回:
            FeatureExtractor: 返回当前模型实例。
        """
        super().train(mode)
        self._freeze()
        return self

    def _freeze(self):
        """
        冻结模型的所有参数，即设置所有参数的 requires_grad 属性为 False。
        """
        super().train(mode=False)
        for p in self.parameters():
            p.requires_grad = False

    @property
    @abstractmethod
    def feature_dims(self) -> List[int]:
        """
        抽象属性，用于获取特征维度列表。

        返回:
            List[int]: 特征维度列表。
        """
        pass

    @property
    @abstractmethod
    def feature_size(self) -> int:
        """
        抽象属性，用于获取特征大小。

        返回:
            int: 特征大小。
        """
        pass

    @property
    @abstractmethod
    def num_groups(self) -> int:
        """
        抽象属性，用于获取特征组的数量。

        返回:
            int: 特征组的数量。
        """
        pass

    @property
    @abstractmethod
    def grouped_indices(self, features):
        """
        抽象属性，用于获取分组索引。

        参数:
            features: 输入的特征。

        返回:
            分组索引。
        """
        pass


def ensemble_logits_with_labels(
    logits: torch.Tensor, labels: List[List[str]], ensemble_method: str = "max"
):
    """
    集成多个模型的logits。

    参数:
        logits (torch.Tensor): 每个模型的logits，最后一个维度是概率。
        labels (list[list[str]]): 标签列表的列表。
        ensemble_method (str): 集成方法，选项为 'mean' 和 'max'，默认为 'max'。

    返回:
        torch.Tensor: 集成模型的logits。
    """
    len_list = [len(l) for l in labels]
    assert logits.shape[-1] == sum(len_list), f"{logits.shape[-1]} != {sum(len_list)}"
    assert ensemble_method in ["mean", "max"]
    ensemble_logits = torch.zeros(
        *logits.shape[:-1], len(labels), dtype=logits.dtype, device=logits.device
    )
    if ensemble_method == "max":
        for i in range(len(labels)):
            ensemble_logits[..., i] = (
                logits[..., sum(len_list[:i]) : sum(len_list[: i + 1])].max(dim=-1).values
            )
    elif ensemble_method == "mean":
        for i in range(len(labels)):
            ensemble_logits[..., i] = logits[..., sum(len_list[:i]) : sum(len_list[: i + 1])].mean(
                dim=-1
            )
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")

    return ensemble_logits


# Ref:https://stackoverflow.com/questions/27049998/convert-a-mixed-nested-list-to-a-nested-tuple
def to_tuple(lst):
    """
    将嵌套列表转换为嵌套元组。

    参数:
        lst: 输入的嵌套列表。

    返回:
        tuple: 转换后的嵌套元组。
    """
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)