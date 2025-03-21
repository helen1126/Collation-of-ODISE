# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import os
from detectron2.data import MetadataCatalog
from mask2former.data.datasets.register_coco_panoptic_annos_semseg import (
    get_metadata,
    register_coco_panoptic_annos_sem_seg,
)

# 预定义的 COCO 全景标注和语义分割标注的数据集分割配置
# 键为数据集名称，值为一个元组，包含全景标注目录、全景标注 JSON 文件和语义分割标注目录
_PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION = {
    "coco_2017_train_panoptic_caption": (
        # 这是原始的全景标注目录
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_caption_train2017.json",
        # 这个目录包含从全景标注转换而来的语义标注
        # 它由 PanopticFPN 使用
        # 你可以使用 detectron2/datasets/prepare_panoptic_fpn.py 脚本创建这些目录
        "coco/panoptic_semseg_train2017",
    ),
    "coco_2017_val_panoptic_caption": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_caption_val2017.json",
        "coco/panoptic_semseg_val2017",
    ),
    "coco_2017_val_100_panoptic_caption": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_caption_val2017_100.json",
        "coco/panoptic_semseg_val2017_100",
    ),
}


# 注册所有 COCO 全景标注和语义分割标注的数据集，并添加 caption 信息
# 注意：数据集名称为 "coco_2017_train_panoptic_caption_with_sem_seg" 和 "coco_2017_val_panoptic_caption_with_sem_seg"
def register_all_coco_panoptic_annos_sem_seg_caption(root):
    """
    注册所有预定义的 COCO 全景标注和语义分割标注数据集，并添加 caption 信息。

    参数:
        root (str): 数据集的根目录，用于拼接各个子目录的完整路径。

    异常:
        ValueError: 如果预定义的数据集名称前缀不是以 "_panoptic_caption" 结尾，会抛出此异常。
    """
    # 遍历预定义的数据集分割配置
    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION.items():
        # 检查数据集名称前缀是否以 "_panoptic_caption" 结尾
        if prefix.endswith("_panoptic_caption"):
            # 提取去除 "_panoptic_caption" 后的前缀
            prefix_instances = prefix[: -len("_panoptic_caption")]
        else:
            # 若前缀不符合要求，抛出异常
            raise ValueError("Unknown prefix: {}".format(prefix))
        # 获取实例数据集的元数据
        instances_meta = MetadataCatalog.get(prefix_instances)
        # 从元数据中获取图像根目录和实例标注 JSON 文件路径
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file

        # 调用外部函数注册 COCO 全景标注和语义分割标注数据集
        register_coco_panoptic_annos_sem_seg(
            prefix,
            get_metadata(),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )


# 调用注册函数，使用环境变量中的数据集根目录，如果未设置则使用默认值 "datasets"
register_all_coco_panoptic_annos_sem_seg_caption(os.getenv("DETECTRON2_DATASETS", "datasets"))