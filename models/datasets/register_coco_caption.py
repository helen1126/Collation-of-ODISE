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

# Ԥ����� COCO ȫ����ע������ָ��ע�����ݼ��ָ�����
# ��Ϊ���ݼ����ƣ�ֵΪһ��Ԫ�飬����ȫ����עĿ¼��ȫ����ע JSON �ļ�������ָ��עĿ¼
_PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION = {
    "coco_2017_train_panoptic_caption": (
        # ����ԭʼ��ȫ����עĿ¼
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_caption_train2017.json",
        # ���Ŀ¼������ȫ����עת�������������ע
        # ���� PanopticFPN ʹ��
        # �����ʹ�� detectron2/datasets/prepare_panoptic_fpn.py �ű�������ЩĿ¼
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


# ע������ COCO ȫ����ע������ָ��ע�����ݼ�������� caption ��Ϣ
# ע�⣺���ݼ�����Ϊ "coco_2017_train_panoptic_caption_with_sem_seg" �� "coco_2017_val_panoptic_caption_with_sem_seg"
def register_all_coco_panoptic_annos_sem_seg_caption(root):
    """
    ע������Ԥ����� COCO ȫ����ע������ָ��ע���ݼ�������� caption ��Ϣ��

    ����:
        root (str): ���ݼ��ĸ�Ŀ¼������ƴ�Ӹ�����Ŀ¼������·����

    �쳣:
        ValueError: ���Ԥ��������ݼ�����ǰ׺������ "_panoptic_caption" ��β�����׳����쳣��
    """
    # ����Ԥ��������ݼ��ָ�����
    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION.items():
        # ������ݼ�����ǰ׺�Ƿ��� "_panoptic_caption" ��β
        if prefix.endswith("_panoptic_caption"):
            # ��ȡȥ�� "_panoptic_caption" ���ǰ׺
            prefix_instances = prefix[: -len("_panoptic_caption")]
        else:
            # ��ǰ׺������Ҫ���׳��쳣
            raise ValueError("Unknown prefix: {}".format(prefix))
        # ��ȡʵ�����ݼ���Ԫ����
        instances_meta = MetadataCatalog.get(prefix_instances)
        # ��Ԫ�����л�ȡͼ���Ŀ¼��ʵ����ע JSON �ļ�·��
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file

        # �����ⲿ����ע�� COCO ȫ����ע������ָ��ע���ݼ�
        register_coco_panoptic_annos_sem_seg(
            prefix,
            get_metadata(),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )


# ����ע�ắ����ʹ�û��������е����ݼ���Ŀ¼�����δ������ʹ��Ĭ��ֵ "datasets"
register_all_coco_panoptic_annos_sem_seg_caption(os.getenv("DETECTRON2_DATASETS", "datasets"))