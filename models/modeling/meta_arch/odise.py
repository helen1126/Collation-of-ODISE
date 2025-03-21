# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import logging
import numpy as np
import operator
from collections import OrderedDict
from typing import Any, Mapping
import diffdist.functional as diff_dist
import torch
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils import comm
from detectron2.utils.memory import retry_if_cuda_oom
from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MLP,
    MultiScaleMaskedTransformerDecoder,
)
from torch import nn
from torch.nn import functional as F

from data.build import get_openseg_labels, prompt_labels

from .clip import ClipAdapter, MaskCLIP, build_clip_text_embed
from .helper import ensemble_logits_with_labels

logger = logging.getLogger(__name__)


# Ref:https://stackoverflow.com/questions/27049998/convert-a-mixed-nested-list-to-a-nested-tuple
def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


@torch.no_grad()
def _concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if comm.get_world_size() == 1:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_world_batch_sizes(batch_size: int, device):
    batch_size = torch.as_tensor([batch_size], dtype=torch.long, device=device)
    global_batch_sizes = _concat_all_gather(batch_size)
    return global_batch_sizes


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors, with dynamic batch size.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(
        max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    padded_tensor[: tensor.shape[0]] = tensor

    tensors_gather = [
        torch.ones((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        for _ in range(comm.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, padded_tensor, async_op=False)

    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])

    output = torch.cat(results, dim=0)
    return output


def dist_collect(tensor):
    """
    Performs all_gather operation on the provided tensors, with dynamic batch size.
    Use diff_dist to get gradient
    """
    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(
        max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    padded_tensor[: tensor.shape[0]] = tensor

    tensors_gather = [
        torch.ones((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        for _ in range(comm.get_world_size())
    ]
    tensors_gather = diff_dist.all_gather(tensors_gather, padded_tensor)

    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])

    output = torch.cat(results, dim=0)
    return output


class ODISE(MaskFormer):
    def ignored_state_dict(self, destination=None, prefix=""):
        """
        获取模型中需要忽略的状态字典。

        参数:
        destination (OrderedDict, 可选): 用于存储忽略状态字典的目标字典。默认为 None。
        prefix (str, 可选): 状态字典键的前缀。默认为空字符串。

        返回:
        OrderedDict: 包含需要忽略的状态字典。
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return destination

    def _open_state_dict(self):
        """
        获取模型的开放状态字典。

        返回:
        dict: 包含模型开放状态的字典，如语义分割头的类别数量、元数据等。
        """
        return {
            "sem_seg_head.num_classes": self.sem_seg_head.num_classes,
            "metadata": self.metadata,
            "test_topk_per_image": self.test_topk_per_image,
            "semantic_on": self.semantic_on,
            "panoptic_on": self.panoptic_on,
            "instance_on": self.instance_on,
        }

    def _save_open_state_dict(self, destination, prefix):
        """
        将模型的开放状态字典保存到指定的目标字典中。

        参数:
        destination (OrderedDict): 用于存储开放状态字典的目标字典。
        prefix (str): 状态字典键的前缀。
        """
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=""):
        """
        获取并保存模型的开放状态字典。

        参数:
        destination (OrderedDict, 可选): 用于存储开放状态字典的目标字典。默认为 None。
        prefix (str, 可选): 状态字典键的前缀。默认为空字符串。

        返回:
        OrderedDict: 包含模型开放状态的字典。
        """
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "open_state_dict"):
                module.open_state_dict(destination, prefix + name + ".")
        return destination

    def load_open_state_dict(self, state_dict: Mapping[str, Any]):
        """
        加载模型的开放状态字典。

        参数:
        state_dict (Mapping[str, Any]): 包含模型开放状态的字典。

        异常:
        AssertionError: 如果状态字典中的某个键值对加载不正确，将抛出此异常。
        """
        for k, v in state_dict.items():
            # handle nested modules
            if len(k.rsplit(".", 1)) == 2:
                prefix, suffix = k.rsplit(".", 1)
                operator.attrgetter(prefix)(self).__setattr__(suffix, v)
            else:
                self.__setattr__(k, v)
            assert operator.attrgetter(k)(self) == v, f"{k} is not loaded correctly"


class CategoryODISE(ODISE):
    def __init__(
        self,
        *,
        category_head=None,
        clip_head=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.category_head = category_head
        self.clip_head = clip_head

    def cal_pred_logits(self, outputs):
        # [B, Q, C]
        mask_embed = outputs["mask_embed"]
        # [K, C]
        text_embed = outputs["text_embed"]
        # [1, C]
        text_embed = outputs["text_embed"]
        null_embed = outputs["null_embed"]

        labels = outputs["labels"]

        mask_embed = F.normalize(mask_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        logit_scale = outputs["logit_scale"]

        # [B, Q, K]
        pred = logit_scale * (mask_embed @ text_embed.t())

        pred = ensemble_logits_with_labels(pred, labels, ensemble_method="max")

        null_embed = F.normalize(null_embed, dim=-1)
        null_pred = logit_scale * (mask_embed @ null_embed.t())

        # [B, Q, K+1]
        pred = torch.cat([pred, null_pred], dim=-1)

        return pred

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the
                        values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        denormalized_images = ImageList.from_tensors(
            [x["image"].to(self.device) / 255.0 for x in batched_inputs]
        )

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        outputs["images"] = denormalized_images.tensor

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            if self.category_head is not None:
                category_head_outputs = self.category_head(outputs, targets)
                outputs.update(category_head_outputs)
                # inplace change pred_logits
                outputs["pred_logits"] = self.cal_pred_logits(outputs)
                if "aux_outputs" in outputs:
                    for aux_outputs in outputs["aux_outputs"]:
                        aux_outputs.update(category_head_outputs)
                        # inplace change pred_logits
                        aux_outputs["pred_logits"] = self.cal_pred_logits(aux_outputs)

            # CLIP head needs output to prepare targets
            # disable for now
            # targets = self.clip_head.prepare_targets(outputs, targets)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:

            # get text_embeddings
            outputs.update(self.category_head(outputs))

            outputs["pred_logits"] = self.cal_pred_logits(outputs)

            mask_pred_results = outputs["pred_masks"]
            mask_cls_results = outputs["pred_logits"]

            if self.clip_head is not None:
                if self.clip_head.with_bg:
                    # [B, Q, K+1]
                    outputs["pred_open_logits"] = outputs["pred_logits"]
                    outputs.update(self.clip_head(outputs))
                    mask_cls_results = outputs["pred_open_logits"]
                else:
                    # [B, Q, K]
                    outputs["pred_open_logits"] = outputs["pred_logits"][..., :-1]
                    outputs.update(self.clip_head(outputs))

                    # merge with bg scores
                    open_logits = outputs["pred_open_logits"]

                    # in case the prediction is not binary
                    binary_probs = torch.zeros(
                        (mask_cls_results.shape[0], mask_cls_results.shape[1], 2),
                        device=mask_cls_results.device,
                        dtype=mask_cls_results.dtype,
                    )
                    binary_probs[..., -1] = F.softmax(mask_cls_results, dim=-1)[..., -1]
                    binary_probs[..., 0] = 1 - binary_probs[..., -1]

                    masks_class_probs = F.softmax(open_logits, dim=-1)
                    # [B, Q, K+1]
                    mask_cls_results = torch.cat(
                        [masks_class_probs * binary_probs[..., 0:1], binary_probs[..., 1:2]], dim=-1
                    )
                    # NOTE: mask_cls_results is already multiplied with logit_scale,
                    # avoid double scale, which cause overflow in softmax
                    # mask_cls_results = torch.log(mask_cls_results + 1e-8) * outputs["logit_scale"]
                    mask_cls_results = torch.log(mask_cls_results + 1e-8)

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["instances"] = instance_r

            return processed_results


class CategoryODISE(ODISE):
    def __init__(
        self,
        *,
        category_head=None,
        clip_head=None,
        **kwargs,
    ):
        pass  # function body is omitted

    def cal_pred_logits(self, outputs):
        # [B, Q, C]
        pass  # function body is omitted

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the
                        values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        pass  # function body is omitted


class CaptionODISE(ODISE):
    def __init__(
        self,
        *,
        word_head=None,
        clip_head=None,
        grounding_criterion=None,
        **kwargs,
    ):
        """
        初始化 CaptionODISE 模型。

        参数:
        word_head (nn.Module, 可选): 用于处理文本信息的模块，默认为 None。
        clip_head (nn.Module, 可选): 基于 CLIP 的模块，默认为 None。
        grounding_criterion (nn.Module, 可选): 用于计算 grounding 损失的准则，默认为 None。
        **kwargs: 传递给父类 ODISE 的其他参数。
        """
        super().__init__(**kwargs)
        self.word_head = word_head
        self.clip_head = clip_head
        self.grounding_criterion = grounding_criterion

    def prepare_targets(self, targets, images):
        """
        准备训练目标，对目标掩码进行填充以匹配图像尺寸。

        参数:
        targets (list): 每个元素为一个图像的目标实例信息。
        images (ImageList): 输入的图像列表。

        返回:
        list: 处理后的目标信息列表，包含填充后的掩码和类别标签。
        """
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )

            if targets_per_image.has("original_gt_classes"):
                # "labels" maybe binary, store original labels in as well
                new_targets[-1]["original_labels"] = targets_per_image.original_gt_classes

        return new_targets

    def prepare_pseudo_targets(self, images):
        """
        准备伪目标，用于没有真实标注的情况。

        参数:
        images (ImageList): 输入的图像列表。

        返回:
        list: 伪目标信息列表，包含全零的掩码和类别标签。
        """
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for _ in range(len(images)):
            # pad gt
            padded_masks = torch.zeros((0, h_pad, w_pad), dtype=torch.bool, device=images.device)
            new_targets.append(
                {
                    "labels": torch.zeros(0, dtype=torch.long, device=images.device),
                    "masks": padded_masks,
                }
            )

        return new_targets

    @property
    def binary_classification(self):
        """
        判断是否为二分类任务。

        返回:
        bool: 如果语义分割头的类别数为 1，则为二分类任务，返回 True；否则返回 False。
        """
        return self.sem_seg_head.num_classes == 1

    def cal_pred_open_logits(self, outputs):
        """
        计算开放词汇的预测 logits。

        参数:
        outputs (dict): 模型的输出，包含掩码嵌入和文本嵌入等信息。

        返回:
        torch.Tensor: 开放词汇的预测 logits，形状为 [B, Q, K]。
        """
        # [B, Q, C]
        mask_embed = outputs["mask_embed"]
        # [K, C]
        text_embed = outputs["text_embed"]

        labels = outputs["labels"]

        mask_embed = F.normalize(mask_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        logit_scale = outputs["logit_scale"]

        # [B, Q, K]
        pred = logit_scale * (mask_embed @ text_embed.t())

        pred = ensemble_logits_with_labels(pred, labels, ensemble_method="max")

        return pred

    def forward(self, batched_inputs):
        """
        前向传播函数，处理输入数据并返回预测结果。

        参数:
        batched_inputs (list): 批量输入数据，每个元素为一个字典，包含图像和目标信息。

        返回:
        list[dict]: 每个字典包含一张图像的预测结果，如语义分割、全景分割或实例分割结果。
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        denormalized_images = ImageList.from_tensors(
            [x["image"].to(self.device) / 255.0 for x in batched_inputs]
        )

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        outputs["images"] = denormalized_images.tensor

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

                if self.binary_classification:
                    # NOTE: convert to binary classification target
                    for i in range(len(gt_instances)):
                        gt_instances[i].original_gt_classes = gt_instances[i].gt_classes.clone()
                        gt_instances[i].gt_classes = torch.zeros_like(gt_instances[i].gt_classes)

                targets = self.prepare_targets(gt_instances, images)
                has_anno = True
            else:
                targets = self.prepare_pseudo_targets(images)
                has_anno = False

            if "captions" in batched_inputs[0]:
                gt_captions = [x["captions"] for x in batched_inputs]
                targets = self.word_head.prepare_targets(gt_captions, targets)

            if self.word_head is not None:
                word_head_outputs = self.word_head(outputs, targets)
                outputs.update(word_head_outputs)
                if "aux_outputs" in outputs:
                    for aux_outputs in outputs["aux_outputs"]:
                        aux_outputs.update(word_head_outputs)

            # CLIP head needs output to prepare targets
            # disable for now
            # targets = self.clip_head.prepare_targets(outputs, targets)

            if self.criterion is not None:
                # bipartite matching-based loss
                losses = self.criterion(outputs, targets)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)

                # multiple by 0 to avoid gradient but make sure the param is used
                if not has_anno:
                    for k in list(losses.keys()):
                        losses[k] *= 0
            else:
                losses = {}

            if self.grounding_criterion is not None:
                grounding_losses = self.grounding_criterion(outputs, targets)
                losses.update(grounding_losses)

            return losses
        else:

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            if "pred_open_logits" not in outputs:
                if self.word_head is not None:
                    outputs.update(self.word_head(outputs))
                outputs["pred_open_logits"] = self.cal_pred_open_logits(outputs)

            if self.clip_head is not None:
                outputs.update(self.clip_head(outputs))

            assert mask_cls_results.shape[-1] == 2 and "pred_open_logits" in outputs

            open_logits = outputs["pred_open_logits"]
            binary_probs = F.softmax(mask_cls_results, dim=-1)
            masks_class_probs = F.softmax(open_logits, dim=-1)
            # [B, Q, K+1]
            mask_cls_results = torch.cat(
                [masks_class_probs * binary_probs[..., 0:1], binary_probs[..., 1:2]], dim=-1
            )
            # NOTE: mask_cls_results is already multiplied with logit_scale,
            # avoid double scale, which cause overflow in softmax
            # mask_cls_results = torch.log(mask_cls_results + 1e-8) * outputs["logit_scale"]
            mask_cls_results = torch.log(mask_cls_results + 1e-8)

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["instances"] = instance_r

            return processed_results



class ODISEMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    def __init__(
        self,
        *,
        class_embed=None,
        mask_embed=None,
        post_mask_embed=None,
        **kwargs,
    ):
        """
        初始化 ODISEMultiScaleMaskedTransformerDecoder 类的实例。

        参数:
        class_embed (nn.Module, 可选): 用于类别嵌入的模块，默认为 None。
        mask_embed (nn.Module, 可选): 用于掩码嵌入的模块，默认为 None。
        post_mask_embed (nn.Module, 可选): 后处理掩码嵌入的模块，默认为 None。
        **kwargs: 传递给父类 MultiScaleMaskedTransformerDecoder 的其他参数。
        """
        super().__init__(**kwargs)
        assert self.mask_classification

        if class_embed is not None:
            self.class_embed = class_embed
        if mask_embed is not None:
            self.mask_embed = mask_embed
        if post_mask_embed is not None:
            assert mask_embed is None
        self.post_mask_embed = post_mask_embed

    def forward(self, x, mask_features, mask=None, *, inputs_dict=None):
        """
        前向传播函数，处理输入特征并生成预测结果。

        参数:
        x (list): 多尺度特征列表，每个元素是一个特征图。
        mask_features (torch.Tensor): 掩码特征张量。
        mask (torch.Tensor, 可选): 掩码张量，默认为 None。
        inputs_dict (dict, 可选): 额外的输入字典，默认为 None。

        返回:
        dict: 包含预测结果的字典，如预测的类别 logits、掩码和辅助输出等。
        """
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(
                self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]
            )

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_extra_results = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, extra_results = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0], inputs_dict=inputs_dict
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_extra_results.append(extra_results)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask, extra_results = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                inputs_dict=inputs_dict,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_extra_results.append(extra_results)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
        }

        # adding extra_results to out and out["aux_outputs"]
        for k in predictions_extra_results[-1].keys():
            out[k] = predictions_extra_results[-1][k]
            for i in range(len(predictions_extra_results) - 1):
                out["aux_outputs"][i][k] = predictions_extra_results[i][k]

        return out

    def forward_prediction_heads(
        self, output, mask_features, attn_mask_target_size, *, inputs_dict=None
    ):
        """
        前向传播预测头函数，根据输入生成类别预测、掩码预测和注意力掩码。

        参数:
        output (torch.Tensor): 解码器的输出张量。
        mask_features (torch.Tensor): 掩码特征张量。
        attn_mask_target_size (tuple): 注意力掩码的目标尺寸。
        inputs_dict (dict, 可选): 额外的输入字典，默认为 None。

        返回:
        tuple: 包含类别预测、掩码预测、注意力掩码和额外结果的元组。
        """
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)

        extra_results = dict()

        mask_embed_results = self.mask_embed(decoder_output)
        if isinstance(mask_embed_results, dict):
            mask_embed = mask_embed_results.pop("mask_embed")
            extra_results.update(mask_embed_results)
        # BC
        else:
            mask_embed = mask_embed_results

        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        if self.post_mask_embed is not None:
            post_mask_embed_results = self.post_mask_embed(
                decoder_output, mask_embed, mask_features, outputs_class, outputs_mask
            )

            if "outputs_mask" in post_mask_embed_results:
                outputs_mask = post_mask_embed_results.pop("outputs_mask")

            extra_results.update(post_mask_embed_results)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(
            outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False
        )
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend,
        # while ``False`` values will be unchanged.
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, extra_results

class MaskGroundingCriterion(nn.Module):
    def __init__(
        self,
        collect_mode: str = "concat",
        loss_weight=1.0,
    ):
        """
        初始化 MaskGroundingCriterion 类的实例。

        参数:
        collect_mode (str, 可选): 收集模式，可取值为 "concat", "diff" 或 None。默认为 "concat"。
        loss_weight (float, 可选): 损失权重，默认为 1.0。
        """
        super().__init__()

        self.collect_mode = collect_mode
        self.loss_weight = loss_weight

        if collect_mode == "diff":
            self.collect_func = dist_collect
        elif collect_mode == "concat":
            self.collect_func = concat_all_gather
        elif collect_mode is None:
            self.collect_func = lambda x: x
        else:
            raise ValueError(f"collect_mode {collect_mode} is not supported")

    def extra_repr(self) -> str:
        """
        返回类的额外表示信息。

        返回:
        str: 包含收集模式和损失权重的字符串。
        """
        return f"collect_mode={self.collect_mode}, \n" f"loss_weight={self.loss_weight} \n"

    def forward(self, outputs, targets):
        """
        前向传播函数，计算损失。

        参数:
        outputs (dict): 模型的输出，包含预测的相关信息。
        targets (list): 目标标签列表，每个元素是一个字典。

        返回:
        dict: 包含计算得到的损失的字典。
        """
        losses = {}
        losses.update(self.get_loss(outputs, targets))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                l_dict = self.get_loss(aux_outputs, targets)
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def get_loss(self, outputs, targets):
        """
        计算掩码与单词嵌入之间的对比损失。

        参数:
        outputs (dict): 模型的输出，包含掩码嵌入、单词嵌入和 logit 缩放因子。
        targets (list): 目标标签列表，每个元素是一个字典，包含单词有效掩码。

        返回:
        dict: 包含计算得到的损失的字典，键为 "loss_mask_word"。
        """
        logit_scale = outputs["logit_scale"]

        rank = comm.get_rank() if self.collect_mode is not None else 0

        # normalized embeds
        # [B, Q, C]
        mask_embed = outputs["mask_embed"]
        # [B, K, C]
        word_embed = outputs["word_embed"]
        # [B, K]
        word_valid_mask = torch.stack([t["word_valid_mask"] for t in targets], dim=0)

        mask_embed = F.normalize(mask_embed, dim=-1)
        word_embed = F.normalize(word_embed, dim=-1)

        batch_size, num_queries, embed_dim = mask_embed.shape
        assert batch_size == word_embed.shape[0], f"{batch_size} != {word_embed.shape[0]}"
        assert embed_dim == word_embed.shape[2], f"{embed_dim} != {word_embed.shape[2]}"
        num_words = word_embed.shape[1]

        # [B*Q, C]
        mask_embed = mask_embed.reshape(batch_size * num_queries, embed_dim)
        # [B*K, C]
        word_embed = word_embed.reshape(batch_size * num_words, embed_dim)

        if self.collect_mode is not None and comm.get_world_size() > 1:
            global_batch_sizes = get_world_batch_sizes(batch_size, device=mask_embed.device)
            global_batch_size = global_batch_sizes.sum().item()
        else:
            global_batch_sizes = None
            global_batch_size = batch_size

        # [W*B*Q, B*K]
        sim_global_mask_word = self.collect_func(mask_embed) @ word_embed.t() * logit_scale

        # [W*B, Q, B, K]
        sim_global_mask_word = sim_global_mask_word.view(
            global_batch_size, num_queries, batch_size, num_words
        )

        # [W*B, B]
        sim_global_img_txt = (
            (sim_global_mask_word.softmax(dim=1) * sim_global_mask_word).sum(dim=1).mean(dim=-1)
        )

        # [B*Q, W*B*K]
        sim_mask_global_word = mask_embed @ self.collect_func(word_embed).t() * logit_scale

        # [B, Q, W*B, K]
        sim_mask_global_word = sim_mask_global_word.view(
            batch_size, num_queries, global_batch_size, num_words
        )

        # [B, W*B]
        sim_img_global_txt = (
            (sim_mask_global_word.softmax(dim=1) * sim_mask_global_word).sum(dim=1).mean(dim=-1)
        )

        if global_batch_sizes is None:
            # get label globally
            labels = (
                torch.arange(batch_size, dtype=torch.long, device=mask_embed.device)
                + batch_size * rank
            )
        else:
            # get label globally and dynamically
            labels = (
                torch.arange(batch_size, dtype=torch.long, device=mask_embed.device)
                + global_batch_sizes[:rank].sum()
            )

        # [B]
        valid_mask = word_valid_mask.any(dim=-1)
        # [W*B]
        global_valid_mask = self.collect_func(valid_mask)

        # [WxB, B] -> [B, WXB] -> [B]
        loss_global_img_txt = F.cross_entropy(sim_global_img_txt.t(), labels, reduction="none")
        loss_global_img_txt = (loss_global_img_txt * valid_mask).mean()

        # [B, WXB] -> [B]
        loss_img_global_txt = F.cross_entropy(
            sim_img_global_txt, labels, weight=global_valid_mask.float()
        )
        if not torch.isfinite(loss_img_global_txt).all():
            # TODO: find reason. Not using vaid mask if NaN as temporary solution
            loss_img_global_txt = F.cross_entropy(sim_img_global_txt, labels)

        loss = 0.5 * (loss_global_img_txt + loss_img_global_txt)

        return {"loss_mask_word": loss * self.loss_weight}


class PseudoClassEmbed(nn.Module):
    def __init__(self, num_classes):
        """
        初始化 PseudoClassEmbed 类的实例。

        参数:
        num_classes (int): 类别的数量。
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        """
        前向传播函数，将输入 x 转换为类别预测的 logits。
        此函数将所有预测视为前景，背景预测为零。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 类别预测的 logits，形状为 (*x.shape[:-1], num_classes + 1)，
                      其中前 num_classes 个维度表示前景类别，最后一个维度表示背景。
        """
        # predict as foreground only
        fg_logits = torch.ones((*x.shape[:-1], self.num_classes), dtype=x.dtype, device=x.device)
        bg_logits = torch.zeros((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
        logits = torch.cat([fg_logits, bg_logits], dim=-1)
        return logits


class MaskPooling(nn.Module):
    def __init__(
        self,
        hard_pooling=True,
        mask_threshold=0.5,
    ):
        """
        初始化 MaskPooling 类的实例。

        参数:
        hard_pooling (bool, 可选): 是否使用硬池化，若为 True 则池化不可微，默认为 True。
        mask_threshold (float, 可选): 掩码阈值，用于硬池化，默认为 0.5。
        """
        super().__init__()
        # if the pooling is hard, it's not differentiable
        self.hard_pooling = hard_pooling
        self.mask_threshold = mask_threshold

    def extra_repr(self) -> str:
        """
        返回类实例的额外信息字符串。

        返回:
        str: 包含硬池化标志和掩码阈值的字符串。
        """
        return f"hard_pooling={self.hard_pooling}\n" f"mask_threshold={self.mask_threshold}\n"

    def forward(self, x, mask):
        """
        前向传播函数，对输入特征图根据掩码进行池化操作。

        参数:
        x (torch.Tensor): 输入特征图，形状为 [B, C, H, W]，其中 B 是批量大小，C 是通道数，H 和 W 分别是高度和宽度。
        mask (torch.Tensor): 掩码张量，形状为 [B, Q, H, W]，其中 Q 是查询数量。

        返回:
        dict: 包含池化后特征的字典，键为 "mask_pooled_features"。
        """

        assert x.shape[-2:] == mask.shape[-2:]

        mask = mask.detach()

        mask = mask.sigmoid()

        if self.hard_pooling:
            mask = (mask > self.mask_threshold).to(mask.dtype)

        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )

        output = {"mask_pooled_features": mask_pooled_x}

        return output


class PooledMaskEmbed(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mask_dim,
        projection_dim,
        temperature=0.07,
    ):
        """
        初始化 PooledMaskEmbed 类的实例。

        参数:
        hidden_dim (int): 隐藏层的维度。
        mask_dim (int): 掩码嵌入的维度。
        projection_dim (int): 投影后的维度。
        temperature (float, 可选): 温度参数，用于缩放 logit 比例，默认为 0.07。
        """
        super().__init__()
        self.pool_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.mask_embed = nn.Sequential(
            nn.LayerNorm(mask_dim), MLP(mask_dim, hidden_dim, projection_dim, 3)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        self.mask_pooling = MaskPooling()

    def forward(self, decoder_output, input_mask_embed, mask_features, pred_logits, pred_masks):
        """
        前向传播函数，对输入进行掩码池化、投影和嵌入操作。

        参数:
        decoder_output (torch.Tensor): 解码器的输出，形状为 [B, Q, C]，其中 B 是批量大小，Q 是查询数量，C 是通道数。
        input_mask_embed (torch.Tensor): 输入的掩码嵌入，形状为 [B, Q, C]。
        mask_features (torch.Tensor): 掩码特征，形状为 [B, C, H, W]，其中 H 和 W 分别是高度和宽度。
        pred_logits (torch.Tensor): 预测的 logits，形状为 [B, Q, K+1]，K 是类别数量。
        pred_masks (torch.Tensor): 预测的掩码，形状为 [B, Q, H, W]。

        返回:
        dict: 包含处理后的结果的字典，键如下：
            - "mask_embed": 掩码嵌入，形状为 [B, Q, projection_dim]。
            - "mask_pooled_features": 池化后的掩码特征，形状为 [B, Q, hidden_dim]。
            - "logit_scale": 缩放后的 logit 比例。
            - "outputs_mask": 如果存在，则为预测的掩码输出。
        """
        mask_pooled_x = self.mask_pooling(mask_features, pred_masks)
        mask_pooled_results = self.mask_pooling(mask_features, pred_masks)
        mask_pooled_x = mask_pooled_results["mask_pooled_features"]
        outputs_mask = mask_pooled_results.get("outputs_mask", None)

        mask_pooled_x = self.pool_proj(mask_pooled_x)

        mask_pooled_x += decoder_output

        mask_embed = self.mask_embed(mask_pooled_x)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)

        output = {
            "mask_embed": mask_embed,
            "mask_pooled_features": mask_pooled_x,
            "logit_scale": logit_scale,
        }

        if outputs_mask is not None:
            output["outputs_mask"] = outputs_mask

        return output


class WordEmbed(nn.Module):
    def __init__(
        self,
        projection_dim,
        clip_model_name="ViT-L-14",
        word_dropout=0.0,
        word_tags="noun_phrase",
        num_words=8,
        prompt="photo",
    ):
        """
        初始化 WordEmbed 类的实例。

        参数:
        projection_dim (int): 投影维度。如果小于 0，则使用 nn.Identity 作为文本投影层。
        clip_model_name (str, 可选): CLIP 模型的名称，默认为 "ViT-L-14"。
        word_dropout (float, 可选): 单词丢弃率，默认为 0.0。
        word_tags (str, 可选): 要提取的单词标签类型，默认为 "noun_phrase"。
        num_words (int, 可选): 每个样本要提取的单词数量，默认为 8。
        prompt (str, 可选): 提示词，默认为 "photo"。
        """
        super().__init__()

        self.clip_model_name = clip_model_name
        self.clip = ClipAdapter(name=self.clip_model_name, normalize=False)

        if projection_dim < 0:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(self.clip.dim_latent, projection_dim)

        self.test_labels = None
        self._test_text_embed_dict = OrderedDict()

        import nltk

        if comm.get_local_rank() == 0:
            nltk.download("popular", quiet=True)
            nltk.download("universal_tagset", quiet=True)
        comm.synchronize()
        self.nltk = nltk

        self.word_dropout = word_dropout
        self.word_tags = word_tags
        self.num_words = num_words
        self.prompt = prompt

    def extra_repr(self) -> str:
        """
        返回类实例的额外信息字符串。

        返回:
        str: 包含 CLIP 模型名称、单词丢弃率、单词标签类型和提取单词数量的字符串。
        """
        return (
            f"clip_model_name={self.clip_model_name},\n"
            f"word_dropout={self.word_dropout},\n"
            f"word_tags={self.word_tags},\n"
            f"num_words={self.num_words}"
        )

    @property
    def device(self):
        """
        获取当前模型所在的设备。

        返回:
        torch.device: 当前模型所在的设备。
        """
        return self.clip.device

    def _open_state_dict(self):
        """
        获取开放状态字典，包含测试标签。

        返回:
        dict: 包含测试标签的字典。
        """
        return {"test_labels": self.test_labels}

    def _save_open_state_dict(self, destination, prefix):
        """
        将开放状态字典保存到目标字典中。

        参数:
        destination (dict): 目标字典，用于保存开放状态。
        prefix (str): 键的前缀。
        """
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=""):
        """
        获取开放状态字典，并将其保存到目标字典中。

        参数:
        destination (dict, 可选): 目标字典，用于保存开放状态。如果为 None，则创建一个新的 OrderedDict。
        prefix (str, 可选): 键的前缀。

        返回:
        dict: 包含开放状态的目标字典。
        """
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "open_state_dict"):
                module.open_state_dict(destination, prefix + name + ".")
        return destination

    @torch.no_grad()
    def build_text_embed(self, labels, verbose=False):
        """
        构建文本嵌入。

        参数:
        labels (list or str): 标签列表或单个标签。
        verbose (bool, 可选): 是否打印详细信息，默认为 False。

        返回:
        torch.Tensor: 文本嵌入张量。
        """
        return build_clip_text_embed(
            clip_model_name=self.clip.clip,
            labels=labels,
            verbose=verbose,
        )

    def get_and_cache_test_text_embed(self, labels):
        """
        获取并缓存测试文本嵌入。

        参数:
        labels (list or str): 标签列表或单个标签。

        返回:
        torch.Tensor: 文本嵌入张量。
        """
        labels = to_tuple(labels)
        if labels not in self._test_text_embed_dict:
            text_embed = self.build_text_embed(labels, verbose=True)
            if len(self._test_text_embed_dict) > 3:
                # pop the first element, only caching 3 elements
                self._test_text_embed_dict.pop(list(self._test_text_embed_dict.keys())[0])
            self._test_text_embed_dict[labels] = text_embed.cpu()
        else:
            text_embed = self._test_text_embed_dict[labels].to(self.device)
        return text_embed

    def get_tag(self, caption, tags):
        """
        从给定的标题中提取指定标签的单词。

        参数:
        caption (str): 标题文本。
        tags (list or str): 要提取的标签列表或单个标签。

        返回:
        list: 提取的单词列表。
        """
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        ret = []
        for (word, pos) in self.nltk.pos_tag(self.nltk.word_tokenize(caption), tagset="universal"):
            for tag in tags:
                if pos == tag:
                    ret.append(word)
        return ret

    def _get_phrase(self, caption, with_preposition):
        """
        从给定的标题中提取名词短语。

        参数:
        caption (str): 标题文本。
        with_preposition (bool): 是否包含介词短语。

        返回:
        list: 提取的名词短语列表。
        """
        if with_preposition:
            # Taken from Su Nam Kim Paper...
            grammar = r"""
                NBAR:
                    {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

                NP:
                    {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                    {<NBAR>} # If pattern is not found, just a single NBAR is ok
            """
        else:
            # Taken from Su Nam Kim Paper...
            grammar = r"""
                NBAR:
                    {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

                NP:
                    {<NBAR>} # If pattern is not found, just a single NBAR is ok
            """
        tokenized = self.nltk.word_tokenize(caption)
        chunker = self.nltk.RegexpParser(grammar)

        chunked = chunker.parse(self.nltk.pos_tag(tokenized))
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if isinstance(subtree, self.nltk.Tree):
                current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk

    def get_noun_phrase(self, caption):
        """
        从给定的标题中提取名词短语，包括有介词和无介词的情况。

        参数:
        caption (str): 标题文本。

        返回:
        list: 提取的名词短语列表。
        """
        noun_phrase = []
        noun_phrase.extend(self._get_phrase(caption, with_preposition=False))
        noun_phrase.extend(self._get_phrase(caption, with_preposition=True))

        return list(set(noun_phrase))

    def prepare_targets(self, captions, targets):
        """
        为目标数据准备单词和有效掩码。

        参数:
        captions (list): 标题列表。
        targets (list): 目标数据列表。

        返回:
        list: 包含准备好的单词和有效掩码的目标数据列表。
        """
        if targets is None:
            targets = [{} for _ in range(len(captions))]

        for caption, target in zip(captions, targets):
            caption = np.random.choice(caption)
            if self.word_tags == "noun_phrase":
                words = self.get_noun_phrase(caption)
            elif "noun_phrase" in self.word_tags:
                words = []
                words.extend(self.get_noun_phrase(caption))
                words.extend(self.get_tag(caption, tuple(set(self.word_tags) - set("noun_phrase"))))
                words = list(set(words))
            else:
                words = self.get_tag(caption, self.word_tags)
            if not len(words):
                words = [""]
            # drop with probability
            words_after_drop = [w for w in words if np.random.rand() > self.word_dropout]
            if len(words_after_drop) == 0:
                # Fall back to no drop if all words are dropped
                words_after_drop = words
            words = np.random.choice(words_after_drop, size=self.num_words).tolist()
            target["words"] = words

            valid_mask = [len(w) > 0 for w in words]
            valid_mask = torch.tensor(valid_mask, device=self.device, dtype=torch.bool)
            target["word_valid_mask"] = valid_mask

        return targets

    def forward(self, outputs, targets=None):
        """
        前向传播函数。

        参数:
        outputs (dict): 模型的输出。
        targets (list, 可选): 目标数据列表。

        返回:
        dict: 包含单词嵌入或文本嵌入的字典。
        """
        if self.training:
            words = [x["words"] for x in targets]

            words = prompt_labels(words, self.prompt)

            word_embed = self.build_text_embed(words)
            # [B, K, C]
            word_embed = torch.stack(word_embed.split([len(w) for w in words]), dim=0)

            word_embed = self.text_proj(word_embed)

            return {"word_embed": word_embed}
        else:
            assert targets is None
            assert self.test_labels is not None
            labels = self.test_labels

            labels = prompt_labels(labels, self.prompt)

            text_embed = self.get_and_cache_test_text_embed(labels)

            text_embed = self.text_proj(text_embed)
            return {"text_embed": text_embed, "labels": labels}


class CategoryEmbed(nn.Module):
    def __init__(
        self,
        labels,
        projection_dim,
        clip_model_name="ViT-L-14",
        prompt=None,
    ):
        """
        初始化 CategoryEmbed 类的实例。

        参数:
        labels (list): 类别标签列表。
        projection_dim (int): 投影维度。如果小于 0，则使用 nn.Identity 作为文本投影层。
        clip_model_name (str, 可选): CLIP 模型的名称，默认为 "ViT-L-14"。
        prompt (str, 可选): 提示词，用于构建文本嵌入，默认为 None。
        """
        super().__init__()
        self.labels = labels

        self.clip_model_name = clip_model_name
        self.clip = ClipAdapter(name=self.clip_model_name, normalize=False)

        if projection_dim < 0:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(self.clip.dim_latent, projection_dim)

        self.register_buffer(
            "text_embed", self.build_text_embed(prompt_labels(labels, prompt), verbose=True), False
        )
        self.null_embed = nn.Parameter(self.build_text_embed(""))

        self.prompt = prompt

        self.test_labels = None
        self._test_text_embed_dict = dict()

    def extra_repr(self) -> str:
        """
        返回类实例的额外信息字符串。

        返回:
        str: 包含 CLIP 模型名称的字符串。
        """
        return f"clip_model_name={self.clip_model_name},\n"

    @property
    def device(self):
        """
        获取当前模型所在的设备。

        返回:
        torch.device: 当前模型所在的设备。
        """
        return self.clip.device

    def _open_state_dict(self):
        """
        获取开放状态字典，包含测试标签。

        返回:
        dict: 包含测试标签的字典。
        """
        return {"test_labels": self.test_labels}

    def _save_open_state_dict(self, destination, prefix):
        """
        将开放状态字典保存到目标字典中。

        参数:
        destination (dict): 目标字典，用于保存开放状态。
        prefix (str): 键的前缀。
        """
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=""):
        """
        获取开放状态字典，并将其保存到目标字典中。

        参数:
        destination (dict, 可选): 目标字典，用于保存开放状态。如果为 None，则创建一个新的 OrderedDict。
        prefix (str, 可选): 键的前缀。

        返回:
        dict: 包含开放状态的目标字典。
        """
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "open_state_dict"):
                module.open_state_dict(destination, prefix + name + ".")
        return destination

    @torch.no_grad()
    def build_text_embed(self, labels, verbose=False):
        """
        构建文本嵌入。

        参数:
        labels (list or str): 标签列表或单个标签。
        verbose (bool, 可选): 是否打印详细信息，默认为 False。

        返回:
        torch.Tensor: 文本嵌入张量。
        """
        return build_clip_text_embed(
            clip_model_name=self.clip.clip,
            labels=labels,
            verbose=verbose,
        )

    def get_and_cache_test_text_embed(self, labels):
        """
        获取并缓存测试文本嵌入。

        参数:
        labels (list or str): 标签列表或单个标签。

        返回:
        torch.Tensor: 文本嵌入张量。
        """
        labels = to_tuple(labels)
        if labels not in self._test_text_embed_dict:
            text_embed = self.build_text_embed(labels, verbose=True)
            self._test_text_embed_dict[labels] = text_embed.cpu()
        else:
            text_embed = self._test_text_embed_dict[labels].to(self.device)
        return text_embed

    def forward(self, outputs, targets=None):
        """
        前向传播函数。

        参数:
        outputs (dict): 模型的输出。
        targets (list, 可选): 目标数据列表。

        返回:
        dict: 包含文本嵌入、空嵌入和标签的字典。
        """
        if self.training:

            text_embed = self.text_proj(self.text_embed)
            null_embed = self.text_proj(self.null_embed)

            return {"text_embed": text_embed, "null_embed": null_embed, "labels": self.labels}

        else:
            assert targets is None
            assert self.test_labels is not None
            labels = self.test_labels
            text_embed = self.get_and_cache_test_text_embed(prompt_labels(labels, self.prompt))

            text_embed = self.text_proj(text_embed)
            null_embed = self.text_proj(self.null_embed)

            return {"text_embed": text_embed, "null_embed": null_embed, "labels": labels}

class CLIPOpenClassEmbed(nn.Module):
    def __init__(
        self,
        labels,
        hidden_dim,
        projection_modality="text",
        clip_model_name="ViT-L-14",
        with_null_embed=True,
        temperature=0.07,
        ensemble_method="max",
    ):
        """
        初始化 CLIPOpenClassEmbed 类的实例。

        参数:
        labels (list): 类别标签列表。
        hidden_dim (int): 隐藏层维度。
        projection_modality (str, 可选): 投影模态，可选值为 "text" 或 "image"，默认为 "text"。
        clip_model_name (str, 可选): CLIP 模型的名称，默认为 "ViT-L-14"。
        with_null_embed (bool, 可选): 是否使用空嵌入，默认为 True。
        temperature (float, 可选): 温度参数，默认为 0.07。
        ensemble_method (str, 可选): 集成方法，可选值为 "max" 或 "mean"，默认为 "max"。
        """
        super().__init__()
        self.labels = labels
        assert projection_modality in ["text", "image"]
        self.projection_modality = projection_modality
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        self.clip_model_name = clip_model_name
        self.register_buffer("text_embed", self.build_text_embed(labels), False)
        if with_null_embed:
            self.null_embed = nn.Parameter(self.build_text_embed(""))
        else:
            self.null_embed = None

        if self.projection_modality == "text":
            self.embed_projection = nn.Linear(self.text_embed.shape[-1], hidden_dim, bias=False)
        else:
            self.embed_projection = nn.Linear(hidden_dim, self.text_embed.shape[-1], bias=False)

        assert ensemble_method in [
            "max",
            "mean",
        ], f"ensemble_method {ensemble_method} not supported"
        self.ensemble_method = ensemble_method

        self.test_labels = None
        self._test_text_embed_dict = dict()

    def _open_state_dict(self):
        """
        获取开放状态字典，包含测试标签。

        返回:
        dict: 包含测试标签的字典。
        """
        return {"test_labels": self.test_labels}

    def _save_open_state_dict(self, destination, prefix):
        """
        将开放状态字典保存到目标字典中。

        参数:
        destination (dict): 目标字典，用于保存开放状态。
        prefix (str): 键的前缀。
        """
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=""):
        """
        获取开放状态字典，并将其保存到目标字典中。

        参数:
        destination (dict, 可选): 目标字典，用于保存开放状态。如果为 None，则创建一个新的 OrderedDict。
        prefix (str, 可选): 键的前缀。

        返回:
        dict: 包含开放状态的目标字典。
        """
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, "open_state_dict"):
                module.open_state_dict(destination, prefix + name + ".")
        return destination

    def extra_repr(self):
        """
        返回类实例的额外信息字符串。

        返回:
        str: 包含 CLIP 模型名称、集成方法和投影模态的字符串。
        """
        return (
            f"clip_model_name={self.clip_model_name}, \n"
            f"ensemble_method={self.ensemble_method}, \n"
            f"projection_modality={self.projection_modality}, \n"
        )

    @torch.no_grad()
    def build_text_embed(self, labels):
        """
        构建文本嵌入。

        参数:
        labels (list or str): 标签列表或单个标签。

        返回:
        torch.Tensor: 文本嵌入张量。
        """
        return build_clip_text_embed(
            clip_model_name=self.clip_model_name,
            labels=labels,
        )

    def get_and_cache_test_text_embed(self, labels):
        """
        获取并缓存测试文本嵌入。

        参数:
        labels (list or str): 标签列表或单个标签。

        返回:
        torch.Tensor: 文本嵌入张量。
        """
        labels = to_tuple(labels)
        if labels not in self._test_text_embed_dict:
            self._test_text_embed_dict[labels] = self.build_text_embed(labels)
        return self._test_text_embed_dict[labels]

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 预测的 logits 张量。
        """
        if self.projection_modality == "image":
            x = self.embed_projection(x)
        x = F.normalize(x, dim=-1)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)

        if self.test_labels is None:
            labels = self.labels
            text_embed = self.text_embed
        else:
            labels = self.test_labels
            text_embed = self.get_and_cache_test_text_embed(labels).to(x.device)

        if self.projection_modality == "text":
            text_embed = self.embed_projection(text_embed)
        text_embed = F.normalize(text_embed, dim=-1)

        # [B, Q, K]
        pred = logit_scale * (x @ text_embed.t())

        pred = ensemble_logits_with_labels(pred, labels, ensemble_method=self.ensemble_method)

        if self.null_embed is not None:
            if self.projection_modality == "text":
                null_embed = self.embed_projection(self.null_embed)
            else:
                null_embed = self.null_embed
            null_embed = F.normalize(null_embed, dim=-1)
            null_pred = logit_scale * (x @ null_embed.t())
            # [B, Q, K+1]
            pred = torch.cat([pred, null_pred], dim=-1)

        return pred


class PoolingCLIPHead(WordEmbed):
    def __init__(
        self,
        clip_model_name="ViT-L-14-336",
        alpha=0.35,
        beta=0.65,
        prompt="photo",
        train_labels=None,
        normalize_logits=True,
        bg_labels=None,
    ):
        """
        初始化 PoolingCLIPHead 类的实例。

        参数:
        clip_model_name (str, 可选): CLIP 模型的名称，默认为 "ViT-L-14-336"。
        alpha (float, 可选): 用于基础类别的融合系数，默认为 0.35。
        beta (float, 可选): 用于新类别的融合系数，默认为 0.65。
        prompt (str, 可选): 提示词，用于构建文本嵌入，默认为 "photo"。
        train_labels (list, 可选): 训练标签列表，默认为 None。
        normalize_logits (bool, 可选): 是否对 logits 进行归一化，默认为 True。
        bg_labels (str, 可选): 背景标签，默认为 None。
        """
        super(WordEmbed, self).__init__()
        self.clip_model_name = clip_model_name
        # For ViT CLIP, we found MaskCLIP yields slightly better performance
        # than pooling on CLIP feature map
        self.clip = MaskCLIP(name=self.clip_model_name)

        self.alpha = alpha
        self.beta = beta

        self.test_labels = None
        self._test_text_embed_dict = dict()

        self.prompt = prompt
        if train_labels is None:
            self.train_labels = get_openseg_labels("coco_panoptic", prompt_engineered=True)
        else:
            self.train_labels = train_labels
        self.bg_labels = bg_labels
        self.normalize_logits = normalize_logits

    def extra_repr(self) -> str:
        """
        返回类实例的额外信息字符串。

        返回:
        str: 包含 CLIP 模型名称的字符串。
        """
        return f"clip_model_name={self.clip_model_name},\n"

    @property
    def with_bg(self):
        """
        判断是否存在背景标签。

        返回:
        bool: 如果存在背景标签则返回 True，否则返回 False。
        """
        return self.bg_labels is not None

    def prepare_targets(self, outputs, targets):
        """
        准备目标数据，获取目标掩码嵌入。

        参数:
        outputs (dict): 模型的输出。
        targets (list): 目标数据列表。

        返回:
        list: 更新后的目标数据列表，包含目标掩码嵌入。
        """
        target_mask_embed = self.clip.get_mask_embed(outputs["images"], outputs["pred_masks"])

        for idx in range(len(targets)):
            targets[idx]["target_mask_embed"] = target_mask_embed[idx]

        return targets

    def forward(self, outputs, targets=None):
        """
        前向传播函数，仅支持推理模式。

        参数:
        outputs (dict): 模型的输出。
        targets (list, 可选): 目标数据列表，默认为 None。

        返回:
        dict: 包含预测的开放类 logits 和标签的字典。
        """
        assert not self.training, "PoolingCLIPHead only supports inference"
        assert targets is None
        assert self.test_labels is not None
        pred_open_logits = outputs.pop("pred_open_logits")

        labels = prompt_labels(self.test_labels, self.prompt)
        if self.with_bg and pred_open_logits.shape[-1] == len(self.test_labels) + 1:
            labels.append(self.bg_labels)

        category_overlapping_list = []

        train_labels = {l for label in self.train_labels for l in label}

        for test_label in self.test_labels:
            category_overlapping_list.append(not set(train_labels).isdisjoint(set(test_label)))

        if self.with_bg and pred_open_logits.shape[-1] == len(self.test_labels) + 1:
            category_overlapping_list.append(False)

        category_overlapping_mask = torch.tensor(
            category_overlapping_list, device=outputs["images"].device, dtype=torch.long
        )

        text_embed = self.get_and_cache_test_text_embed(labels)

        mask_pred_results = outputs["pred_masks"]

        clip_results = self.clip(
            outputs["images"],
            mask_pred_results,
            text_embed,
            labels,
        )

        mask_pred_open_logits = clip_results["mask_pred_open_logits"]

        if self.normalize_logits:
            pred_open_prob = pred_open_logits.softmax(dim=-1)

            mask_pred_open_prob = mask_pred_open_logits.softmax(dim=-1)

            # NOTE: logits are multiplied with logit_scale,
            # avoid double scale, which cause overflow in softmax
            pred_open_logits_base = (
                (pred_open_prob ** (1 - self.alpha) * mask_pred_open_prob**self.alpha).log()
                # * outputs["logit_scale"]
                * category_overlapping_mask
            )

            pred_open_logits_novel = (
                (pred_open_prob ** (1 - self.beta) * mask_pred_open_prob**self.beta).log()
                # * outputs["logit_scale"]
                * (1 - category_overlapping_mask)
            )
        else:

            # NOTE: this version ignore the scale difference during ensemble,

            pred_open_logits_base = (
                pred_open_logits * (1 - self.alpha)
                + mask_pred_open_logits * self.alpha * category_overlapping_mask
            )
            pred_open_logits_novel = pred_open_logits * (
                1 - self.beta
            ) + mask_pred_open_logits * self.beta * (1 - category_overlapping_mask)

        pred_open_logits = pred_open_logits_base + pred_open_logits_novel

        ret = {"pred_open_logits": pred_open_logits}
        if "labels" in outputs:
            ret["labels"] = labels

        return ret
