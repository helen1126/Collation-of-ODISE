# ------------------------------------------------------------------------------
# Copyright (c) 2021 OpenAI
# To view a copy of this license, visit
# https://github.com/openai/glide-text2im/blob/main/LICENSE
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

import numpy as np
from abc import ABC, abstractmethod
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    根据给定的名称和扩散对象，从预定义的采样器库中创建一个 ScheduleSampler 实例。

    参数:
        name (str): 采样器的名称。支持的名称有 "uniform" 和 "loss-second-moment"。
        diffusion: 用于采样的扩散对象。

    返回:
        ScheduleSampler: 创建的采样器实例。

    异常:
        NotImplementedError: 如果提供的名称不是预定义的采样器名称，则抛出此异常。
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    一个抽象基类，代表扩散过程中时间步的分布采样器，旨在减少目标函数的方差。

    默认情况下，采样器执行无偏重要性采样，即目标函数的均值保持不变。
    但是，子类可以重写 sample() 方法来改变重采样项的加权方式，从而改变目标函数。
    """

    @abstractmethod
    def weights(self):
        """
        获取一个 numpy 数组，其中每个元素对应一个扩散步骤的权重。

        权重不需要归一化，但必须为正数。

        返回:
            numpy.ndarray: 包含每个扩散步骤权重的数组。
        """

    def sample(self, batch_size, device):
        """
        为一个批次的样本进行重要性采样时间步。

        参数:
            batch_size (int): 要采样的时间步数量。
            device: 用于保存采样结果的 torch 设备。

        返回:
            tuple: 一个包含两个元素的元组 (timesteps, weights)
                - timesteps (torch.Tensor): 采样得到的时间步索引的张量。
                - weights (torch.Tensor): 用于缩放最终损失的权重张量。
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        """
        初始化均匀采样器。

        参数:
            diffusion: 用于采样的扩散对象。
        """
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        """
        获取均匀采样器的权重。

        返回:
            numpy.ndarray: 包含每个扩散步骤权重的数组，所有权重都为 1。
        """
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        使用模型的局部损失更新重加权。

        每个进程都应该调用此方法，传入一批时间步和对应的损失。
        此方法会进行同步，确保所有进程保持完全相同的重加权。

        参数:
            local_ts (torch.Tensor): 一个整数张量，包含局部时间步。
            local_losses (torch.Tensor): 一个一维张量，包含每个时间步对应的损失。
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        使用模型的损失更新重加权。

        子类应该重写此方法，以使用模型的损失更新重加权。
        此方法直接更新重加权，而不进行进程间的同步。它由 update_with_local_losses 方法从所有进程调用，传入相同的参数。
        因此，它应该具有确定性的行为，以确保所有进程的状态一致。

        参数:
            ts (list): 一个整数列表，包含时间步。
            losses (list): 一个浮点数列表，包含每个时间步对应的损失。
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        """
        初始化基于损失二阶矩的重采样器。

        参数:
            diffusion: 用于采样的扩散对象。
            history_per_term (int, 可选): 每个时间步保存的损失历史数量。默认为 10。
            uniform_prob (float, 可选): 均匀采样的概率。默认为 0.001。
        """
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([diffusion.num_timesteps, history_per_term], dtype=np.float64)
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        """
        获取基于损失二阶矩的重采样器的权重。

        如果损失历史尚未填满，则返回均匀权重。
        否则，计算损失的二阶矩，并结合均匀采样概率计算权重。

        返回:
            numpy.ndarray: 包含每个扩散步骤权重的数组。
        """
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        """
        使用所有损失更新损失历史。

        如果某个时间步的损失历史已满，则移除最旧的损失项。
        否则，将新的损失添加到损失历史中。

        参数:
            ts (list): 一个整数列表，包含时间步。
            losses (list): 一个浮点数列表，包含每个时间步对应的损失。
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """
        检查损失历史是否已经填满。

        返回:
            bool: 如果所有时间步的损失历史都已填满，则返回 True；否则返回 False。
        """
        return (self._loss_counts == self.history_per_term).all()