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
    ���������������ƶ���ָ�����豸�ϡ�

    �ú����ݹ�ش���ͬ���͵������������ݣ�����������ӳ�䣨���ֵ䣩�����У����б���
    �������ƶ���ָ�����豸����GPU���ϡ�����ָ��Ҫ�ų��ļ�����Щ����Ӧ�����ݽ����ᱻ�ƶ���

    ����:
    batched_inputs (torch.Tensor or collections.abc.Mapping or collections.abc.Sequence or str):
        �����������ݣ�������������ӳ�䣨���ֵ䣩�����У����б����ַ�����
    device (torch.device):
        Ŀ���豸�����ݽ����ƶ������豸�ϡ�
    exclude (str or tuple of str, ��ѡ):
        Ҫ�ų��ļ����б�򵥸�������Ӧ��Щ�������ݽ����ᱻ�ƶ���Ŀ���豸��Ĭ��Ϊ��Ԫ�顣

    ����:
    torch.Tensor or collections.abc.Mapping or collections.abc.Sequence or str:
        �ƶ���ָ���豸��������������ݡ�

    �쳣:
    TypeError:
        ����������ݵ����Ͳ���֧�֣����׳����쳣��
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