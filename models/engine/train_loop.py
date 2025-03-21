# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# To view a copy of this license, visit
# https://github.com/facebookresearch/detectron2/blob/main/LICENSE
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import itertools
import logging
import numpy as np
import time
from typing import Iterable, Mapping, Union
import detectron2.utils.comm as comm
import torch
from detectron2.engine import SimpleTrainer as _SimpleTrainer
from detectron2.utils.events import get_event_storage
from torch._six import inf
from torch.nn.parallel import DataParallel, DistributedDataParallel

from utils.parameter_count import parameter_count_table

logger = logging.getLogger(__name__)

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def get_grad_norm(parameters: _tensor_or_tensors, norm_type: float = 2.0) -> torch.Tensor:
    r"""
    计算可迭代参数的梯度范数。
    该范数是将所有梯度视为一个单一向量来计算的。

    参数:
        parameters (Iterable[Tensor] or Tensor): 一个可迭代的张量或单个张量，其梯度将被归一化。
        norm_type (float or int): 所使用的 p 范数类型。可以是 'inf' 表示无穷范数。

    返回:
        torch.Tensor: 参数的总梯度范数（视为一个单一向量）。
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )

    return total_norm


class SimpleTrainer(_SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, *, grad_clip=None):
        """
        初始化 SimpleTrainer 类。

        参数:
            model: 要训练的模型。
            data_loader: 数据加载器，用于提供训练数据。
            optimizer: 优化器，用于更新模型参数。
            grad_clip (可选): 梯度裁剪的阈值。
        """
        super().__init__(model, data_loader, optimizer)
        self.grad_clip = grad_clip
        logger.info(f"Trainer: {self.__class__.__name__}")
        logger.info(f"grad_clip: {grad_clip}")
        logger.info("All parameters: \n" + parameter_count_table(model))

        # 打印可训练参数
        logger.info("Trainable parameters: \n" + parameter_count_table(model, trainable_only=True))

    def raise_loss_nan(self, losses):
        """
        检查损失是否为 NaN 或无穷大，如果是则抛出异常。

        参数:
            losses (torch.Tensor): 损失张量。
        """
        losses = losses.detach().cpu()
        loss_nan = (~torch.isfinite(losses)).any()
        all_loss_nan = comm.all_gather(loss_nan)
        all_loss_nan = [l.item() for l in all_loss_nan]
        if any(all_loss_nan):
            raise FloatingPointError(
                f"Loss became infinite or NaN for rank: {np.where(all_loss_nan)[0].tolist()} "
                f"at iteration={self.storage.iter}!\n"
            )

    def run_step(self):
        """
        实现标准的训练逻辑。
        此方法执行以下步骤：
        1. 从数据加载器中获取数据。
        2. 前向传播计算损失。
        3. 反向传播计算梯度。
        4. 可选地进行梯度裁剪。
        5. 记录梯度范数。
        6. 记录损失指标。
        7. 更新模型参数。
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        如果需要对数据进行处理，可以包装数据加载器。
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        如果需要对损失进行处理，可以包装模型。
        """
        # 注意：如果输入是字典，添加 "runner_meta"，由 Jiarui 添加
        if isinstance(data, dict):
            data["runner_meta"] = dict()
            data["runner_meta"]["iter"] = self.iter
            data["runner_meta"]["max_iter"] = self.max_iter
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        如果需要累积梯度或进行类似操作，可以包装优化器的 zero_grad() 方法。
        """
        self.optimizer.zero_grad()
        losses.backward()

        # 注意：添加 "grad_norm"，由 Jiarui 添加
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        else:
            grad_norm = get_grad_norm(self.model.parameters())
        self.storage.put_scalar("grad_norm", grad_norm)

        # 禁用此检查，查看是否必要
        # 在记录指标之前检查 NaN，退出所有进程
        # self.raise_loss_nan(losses)

        self._write_metrics(loss_dict, data_time)

        """
        如果需要梯度裁剪/缩放或其他处理，可以包装优化器的 step() 方法。
        但如 https://arxiv.org/abs/2006.15704 Sec 3.2.4 所述，这种方法不是最优的。
        """
        self.optimizer.step()

    # 与父类几乎相同，除了记录所有 `metric_dict` 而不是
    # 原始 detectron2 中的 len(metric_dict) > 1，可能是个 bug
    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        """
        记录损失指标。

        参数:
            loss_dict (dict): 标量损失的字典。
            data_time (float): 数据加载器迭代所花费的时间。
            prefix (str, 可选): 日志键的前缀。
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # 在所有工作进程中收集指标以进行日志记录
        # 这假设我们使用 DDP 风格的训练，这是目前 detectron2 唯一支持的方法。
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # 工作进程之间的数据加载时间可能有很大差异。
            # 实际由数据加载时间引起的延迟是所有工作进程中的最大值。
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # 对其余指标求平均值
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict):
                storage.put_scalars(**metrics_dict)


class NativeScalerWithGradNormCount:
    """
    参考: https://github.com/microsoft/Swin-Transformer/blob/afeb877fba1139dfbc186276983af2abb02c2196/main.py#L194
    """  # noqa

    state_dict_key = "amp_scaler"

    def __init__(self):
        """
        初始化 NativeScalerWithGradNormCount 类。
        创建一个 PyTorch 的 GradScaler 用于自动缩放梯度。
        """
        from torch.cuda.amp import GradScaler

        self._scaler = GradScaler()

    def __call__(
        self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True
    ):
        """
        执行梯度缩放、反向传播、梯度裁剪和优化器更新操作。

        参数:
            loss (torch.Tensor): 损失张量。
            optimizer: 优化器。
            clip_grad (可选): 梯度裁剪的阈值。
            parameters (可选): 要计算梯度范数的参数。
            create_graph (bool, 可选): 是否创建计算图，用于二阶导数计算。
            update_grad (bool, 可选): 是否更新梯度。

        返回:
            torch.Tensor: 梯度范数。
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # 就地取消优化器参数的梯度缩放
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm(parameters)

            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        """
        获取 GradScaler 的状态字典。

        返回:
            dict: GradScaler 的状态字典。
        """
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """
        加载 GradScaler 的状态字典。

        参数:
            state_dict (dict): GradScaler 的状态字典。
        """
        self._scaler.load_state_dict(state_dict)


def get_optimizer_parameters(optimizer):
    """
    获取优化器中的所有参数。

    参数:
        optimizer: 优化器。

    返回:
        itertools.chain: 优化器中的所有参数的迭代器。
    """
    return itertools.chain(*[x["params"] for x in optimizer.param_groups])


class AMPTrainer(SimpleTrainer):
    """
    类似于 `SimpleTrainer`，但在训练循环中使用 PyTorch 的原生自动混合精度（AMP）。
    """

    def __init__(self, model, data_loader, optimizer, grad_clip=None):
        """
        初始化 AMPTrainer 类。

        参数:
            model: 要训练的模型。
            data_loader: 数据加载器，用于提供训练数据。
            optimizer: 优化器，用于更新模型参数。
            grad_clip (可选): 梯度裁剪的阈值。
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        # 注意：Jiarui 放宽了此检查，因为对于 AMP 来说不是必需的
        elif isinstance(model, DataParallel):
            assert not len(model.device_ids) > 1, unsupported
        # assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer, grad_clip=grad_clip)

        self.grad_scaler = NativeScalerWithGradNormCount()

    def run_step(self):
        """
        实现 AMP 训练逻辑。
        此方法执行以下步骤：
        1. 从数据加载器中获取数据。
        2. 使用自动混合精度进行前向传播计算损失。
        3. 清零模型参数的梯度。
        4. 使用 GradScaler 进行梯度缩放、反向传播、梯度裁剪和优化器更新。
        5. 记录梯度范数和损失缩放值。
        6. 记录损失指标。
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # detectron2 使用简单的 collate 函数，所以数据只是一个字典列表
        # 我们不为它们添加元信息
        # 对于 webdataset，数据是一个张量字典
        if isinstance(data, dict):
            # 注意：添加 "runner_meta"，由 Jiarui 添加
            data["runner_meta"] = dict()
            data["runner_meta"]["iter"] = self.iter
            data["runner_meta"]["max_iter"] = self.max_iter
        with autocast():
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        # 不是使用 self.optimizer.zero_grad()，
        # 而是将模型中所有参数的梯度清零
        self.model.zero_grad()

        # 注意：以下操作封装在 NativeScalerWithGradNormCount() 的 __call__() 中：
        # self.grad_scaler.scale(losses).backward()
        # self.grad_scaler.step(self.optimizer)
        # self.grad_scaler.update()
        grad_norm = self.grad_scaler(
            losses,
            self.optimizer,
            clip_grad=self.grad_clip,
            # 不是使用 self.model.parameters()，而是使用优化器中的参数
            # 这样我们就不会裁剪不在优化器中的参数
            parameters=get_optimizer_parameters(self.optimizer),
            update_grad=True,
        )
        self.storage.put_scalar("grad_norm", grad_norm)
        self.storage.put_scalar(
            "clipped_grad_norm", get_grad_norm(get_optimizer_parameters(self.optimizer))
        )

        loss_scale_value = self.grad_scaler.state_dict()["scale"]
        self.storage.put_scalar("loss_scale", loss_scale_value)

        self._write_metrics(loss_dict, data_time)

    def state_dict(self):
        """
        获取训练器的状态字典，包括 GradScaler 的状态。

        返回:
            dict: 训练器的状态字典。
        """
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        """
        加载训练器的状态字典，包括 GradScaler 的状态。

        参数:
            state_dict (dict): 训练器的状态字典。
        """
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])