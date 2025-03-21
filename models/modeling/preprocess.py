# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import collections.abc
import torch


def batched_input_to_device(batched_inputs, device, exclude=()):
    """
    将批量输入数据移动到指定的设备上。

    该函数递归地处理不同类型的批量输入数据，包括张量、映射（如字典）和序列（如列表），
    并将其移动到指定的设备（如GPU）上。可以指定要排除的键，这些键对应的数据将不会被移动。

    参数:
    batched_inputs (torch.Tensor or collections.abc.Mapping or collections.abc.Sequence or str):
        批量输入数据，可以是张量、映射（如字典）、序列（如列表）或字符串。
    device (torch.device):
        目标设备，数据将被移动到该设备上。
    exclude (str or tuple of str, 可选):
        要排除的键的列表或单个键。对应这些键的数据将不会被移动到目标设备。默认为空元组。

    返回:
    torch.Tensor or collections.abc.Mapping or collections.abc.Sequence or str:
        移动到指定设备后的批量输入数据。

    异常:
    TypeError:
        如果输入数据的类型不被支持，将抛出此异常。
    """
    if isinstance(exclude, str):
        exclude = [exclude]

    if isinstance(batched_inputs, torch.Tensor):
        batch = batched_inputs.to(device, non_blocking=True)
        return batch
    elif isinstance(batched_inputs, collections.abc.Mapping):
        batch = {}
        for k in batched_inputs:
            if k not in exclude:
                batched_inputs[k] = batched_input_to_device(batched_inputs[k], device)
        return batched_inputs

    elif isinstance(batched_inputs, collections.abc.Sequence) and not isinstance(
        batched_inputs, str
    ):
        return [batched_input_to_device(d, device) for d in batched_inputs]
    elif isinstance(batched_inputs, str):
        return batched_inputs
    else:
        raise TypeError(f"Unsupported type {type(batched_inputs)}")