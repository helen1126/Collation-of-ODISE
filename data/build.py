# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import copy
import logging
import os.path as osp
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.utils import comm


def get_openseg_labels(dataset, prompt_engineered=False):
    """
    获取指定数据集的标签，以双重列表格式返回。
    例如：[[background, bag, bed, ...], ["aeroplane"], ...]

    参数:
    dataset (str): 数据集名称，必须是以下之一：
        "ade20k_150", "ade20k_847", "coco_panoptic",
        "pascal_context_59", "pascal_context_459",
        "pascal_voc_21", "lvis_1203"
    prompt_engineered (bool): 是否使用经过提示工程处理的标签文件，默认为 False

    返回:
    list: 双重列表格式的标签，每个子列表包含一个类别的标签名称
    """

    invalid_name = "invalid_class_id"
    assert dataset in [
        "ade20k_150",
        "ade20k_847",
        "coco_panoptic",
        "pascal_context_59",
        "pascal_context_459",
        "pascal_voc_21",
        "lvis_1203",
    ]

    label_path = osp.join(
        osp.dirname(osp.abspath(__file__)),
        "datasets/openseg_labels",
        f"{dataset}_with_prompt_eng.txt" if prompt_engineered else f"{dataset}.txt",
    )

    # 以 id:name 格式读取文本
    with open(label_path, "r") as f:
        lines = f.read().splitlines()

    categories = []
    for line in lines:
        id, name = line.split(":")
        if name == invalid_name:
            continue
        categories.append({"id": int(id), "name": name})

    return [dic["name"].split(",") for dic in categories]


def prompt_labels(labels, prompt):
    """
    根据指定的提示格式对标签进行处理。

    参数:
    labels (list): 双重列表格式的标签，每个子列表包含一个类别的标签名称
    prompt (str or None): 提示格式，必须是 "a", "photo", "scene" 之一，若为 None 则不进行处理

    返回:
    list: 处理后的双重列表格式的标签
    """
    if prompt is None:
        return labels
    labels = copy.deepcopy(labels)
    assert prompt in ["a", "photo", "scene"]
    if prompt == "a":
        for i in range(len(labels)):
            labels[i] = [f"a {l}" for l in labels[i]]
    elif prompt == "photo":
        for i in range(len(labels)):
            labels[i] = [f"a photo of a {l}." for l in labels[i]]
    elif prompt == "scene":
        for i in range(len(labels)):
            labels[i] = [f"a photo of a {l} in the scene." for l in labels[i]]
    else:
        raise NotImplementedError

    return labels


def build_d2_train_dataloader(
    dataset,
    mapper=None,
    total_batch_size=None,
    local_batch_size=None,
    num_workers=0,
    sampler=None,
):
    """
    构建 Detectron2 的训练数据加载器。

    参数:
    dataset: 数据集对象
    mapper: 数据映射器，用于将数据集样本转换为模型可接受的格式，默认为 None
    total_batch_size (int or None): 全局总批量大小，若指定则必须能被 GPU 数量整除
    local_batch_size (int or None): 本地批量大小，即每个 GPU 上的批量大小
    num_workers (int): 数据加载的工作线程数，默认为 0
    sampler: 数据采样器，默认为 None

    返回:
    DataLoader: Detectron2 的训练数据加载器
    """

    assert (total_batch_size is None) != (
        local_batch_size is None
    ), "Either total_batch_size or local_batch_size must be specified"

    world_size = comm.get_world_size()

    if total_batch_size is not None:
        assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
        ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
            total_batch_size, world_size
        )
        batch_size = total_batch_size // world_size

    if local_batch_size is not None:
        batch_size = local_batch_size

    total_batch_size = batch_size * world_size

    return build_detection_train_loader(
        dataset=dataset,
        mapper=mapper,
        sampler=sampler,
        total_batch_size=total_batch_size,
        aspect_ratio_grouping=True,
        num_workers=num_workers,
        collate_fn=None,
    )


def build_d2_test_dataloader(
    dataset,
    mapper=None,
    total_batch_size=None,
    local_batch_size=None,
    num_workers=0,
):
    """
    构建 Detectron2 的测试数据加载器。

    参数:
    dataset: 数据集对象
    mapper: 数据映射器，用于将数据集样本转换为模型可接受的格式，默认为 None
    total_batch_size (int or None): 全局总批量大小，若指定则必须能被 GPU 数量整除
    local_batch_size (int or None): 本地批量大小，即每个 GPU 上的批量大小
    num_workers (int): 数据加载的工作线程数，默认为 0

    返回:
    DataLoader: Detectron2 的测试数据加载器
    """

    assert (total_batch_size is None) != (
        local_batch_size is None
    ), "Either total_batch_size or local_batch_size must be specified"

    world_size = comm.get_world_size()

    if total_batch_size is not None:
        assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
        ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
            total_batch_size, world_size
        )
        batch_size = total_batch_size // world_size

    if local_batch_size is not None:
        batch_size = local_batch_size

    logger = logging.getLogger(__name__)
    if batch_size != 1:
        logger.warning(
            "When testing, batch size is set to 1. "
            "This is the only mode that is supported for d2."
        )

    return build_detection_test_loader(
        dataset=dataset,
        mapper=mapper,
        sampler=None,
        num_workers=num_workers,
        collate_fn=None,
    )