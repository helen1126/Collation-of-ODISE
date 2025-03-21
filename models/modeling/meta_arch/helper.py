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
        ��ʼ��LayerNorm2d���ʵ����

        ����:
            num_channels (int): ����������ͨ������
            eps (float, optional): Ϊ����ֵ�ȶ�����ӵ���ĸ��һ��С������Ĭ��Ϊ1e-6��
            affine (bool, optional): �Ƿ�ʹ�ÿ�ѧϰ�ķ���任������Ȩ�غ�ƫ�ã���Ĭ��ΪTrue��
        """
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ǰ�򴫲��������������������в��һ��������

        ����:
            x (torch.Tensor): �����2D�ռ���������״Ϊ [N, C, H, W]��

        ����:
            torch.Tensor: �������һ�������������״��������ͬ��
        """
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)


class FeatureExtractor(nn.Module, metaclass=ABCMeta):
    """
    FeatureExtractor ��һ��������࣬���ڶ���������ȡ���Ļ����ӿڡ�
    ���̳��� nn.Module����ʹ�� ABCMeta Ԫ����������󷽷���
    """

    def __init__(self):
        """
        ��ʼ�� FeatureExtractor ���ʵ����
        """
        super().__init__()

    def ignored_state_dict(self, destination=None, prefix=""):
        """
        ��ȡģ�͵�״̬�ֵ䣬����ĳЩģ���״̬��

        ����:
            destination (OrderedDict, optional): ���ڴ洢״̬�ֵ��Ŀ���ֵ䣬Ĭ��Ϊ None��
            prefix (str, optional): ״̬�ֵ��м���ǰ׺��Ĭ��Ϊ���ַ�����

        ����:
            OrderedDict: ����ģ��״̬���ֵ䡣
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
        ��д state_dict ����������һ���յ� OrderedDict��������ģ��״̬��

        ����:
            destination (OrderedDict, optional): ���ڴ洢״̬�ֵ��Ŀ���ֵ䣬Ĭ��Ϊ None��
            prefix (str, optional): ״̬�ֵ��м���ǰ׺��Ĭ��Ϊ���ַ�����
            keep_vars (bool, optional): �Ƿ���������Ĭ��Ϊ False��

        ����:
            OrderedDict: һ���յ� OrderedDict��
        """
        return OrderedDict()

    def train(self, mode: bool = True):
        """
        ����ģ�͵�ѵ��ģʽ��������ģ�Ͳ�����

        ����:
            mode (bool, optional): �Ƿ�Ϊѵ��ģʽ��Ĭ��Ϊ True��

        ����:
            FeatureExtractor: ���ص�ǰģ��ʵ����
        """
        super().train(mode)
        self._freeze()
        return self

    def _freeze(self):
        """
        ����ģ�͵����в��������������в����� requires_grad ����Ϊ False��
        """
        super().train(mode=False)
        for p in self.parameters():
            p.requires_grad = False

    @property
    @abstractmethod
    def feature_dims(self) -> List[int]:
        """
        �������ԣ����ڻ�ȡ����ά���б�

        ����:
            List[int]: ����ά���б�
        """
        pass

    @property
    @abstractmethod
    def feature_size(self) -> int:
        """
        �������ԣ����ڻ�ȡ������С��

        ����:
            int: ������С��
        """
        pass

    @property
    @abstractmethod
    def num_groups(self) -> int:
        """
        �������ԣ����ڻ�ȡ�������������

        ����:
            int: �������������
        """
        pass

    @property
    @abstractmethod
    def grouped_indices(self, features):
        """
        �������ԣ����ڻ�ȡ����������

        ����:
            features: �����������

        ����:
            ����������
        """
        pass


def ensemble_logits_with_labels(
    logits: torch.Tensor, labels: List[List[str]], ensemble_method: str = "max"
):
    """
    ���ɶ��ģ�͵�logits��

    ����:
        logits (torch.Tensor): ÿ��ģ�͵�logits�����һ��ά���Ǹ��ʡ�
        labels (list[list[str]]): ��ǩ�б���б�
        ensemble_method (str): ���ɷ�����ѡ��Ϊ 'mean' �� 'max'��Ĭ��Ϊ 'max'��

    ����:
        torch.Tensor: ����ģ�͵�logits��
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
    ��Ƕ���б�ת��ΪǶ��Ԫ�顣

    ����:
        lst: �����Ƕ���б�

    ����:
        tuple: ת�����Ƕ��Ԫ�顣
    """
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)