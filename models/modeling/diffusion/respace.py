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
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    从原始的扩散过程中创建一个要使用的时间步列表，根据我们希望从原始过程的等大小部分中选取的时间步数来确定。

    例如，如果原始扩散过程有300个时间步，且部分计数为 [10, 15, 20]，
    那么前100个时间步将被步长化为10个时间步，第二个100个时间步将被步长化为15个时间步，
    最后100个时间步将被步长化为20个时间步。

    如果步长是一个以 "ddim" 开头的字符串，则使用DDIM论文中的固定步长，并且只允许一个部分。

    参数:
        num_timesteps (int): 原始扩散过程中要划分的扩散步数。
        section_counts (list or str): 一个数字列表，或者一个包含逗号分隔数字的字符串，
                                      表示每个部分的步数。特殊情况下，使用 "ddimN"，其中N是
                                      要使用的DDIM论文中的步数。

    返回:
        set: 一个来自原始扩散过程的时间步集合，用于后续的扩散过程。
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            # 提取期望的步数
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"无法使用整数步长精确创建 {num_timesteps} 个步骤")
        elif section_counts.startswith("ldm_ddim"):
            # 提取期望的步数
            desired_count = int(section_counts[len("ldm_ddim") :])
            # 与ddim相比，加1以确保采样期间最终的alpha值正确
            # 参考: https://github.com/CompVis/stable-diffusion/blob/d39f5b51a8d607fd855425a0d546b9f871034c3d/ldm/modules/diffusionmodules/util.py#L56
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(1, num_timesteps + 1, i))
            raise ValueError(f"无法使用整数步长精确创建 {num_timesteps} 个步骤")
        elif section_counts == "fast27":
            # 递归调用 space_timesteps 函数
            steps = space_timesteps(num_timesteps, "10,10,3,2,2")
            # 帮助减少最嘈杂时间步的DDIM伪影
            steps.remove(num_timesteps - 1)
            steps.add(num_timesteps - 3)
            return steps
        # 将字符串转换为整数列表
        section_counts = [int(x) for x in section_counts.split(",")]
    # 计算每个部分的大小
    size_per = num_timesteps // len(section_counts)
    # 计算剩余的步数
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        # 计算当前部分的实际大小
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"无法将 {size} 个步骤的部分划分为 {section_count} 个步骤")
        if section_count <= 1:
            frac_stride = 1
        else:
            # 计算分数步长
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    一个可以在基础扩散过程中跳过步骤的扩散过程。

    参数:
        use_timesteps (collection): 一个来自原始扩散过程的时间步集合（序列或集合），表示要保留的时间步。
        **kwargs: 用于创建基础扩散过程的关键字参数。
    """

    def __init__(self, use_timesteps, **kwargs):
        # 将使用的时间步转换为集合
        self.use_timesteps = set(use_timesteps)
        # 存储时间步映射
        self.timestep_map = []
        # 存储原始的步数
        self.original_num_steps = len(kwargs["betas"])

        # 创建基础扩散过程
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                # 计算新的beta值
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                # 记录时间步映射
                self.timestep_map.append(i)
        # 更新betas参数
        kwargs["betas"] = np.array(new_betas)
        # 调用父类的构造函数
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        """
        计算给定模型下的均值和方差。

        参数:
            model: 用于计算的模型。
            *args: 位置参数。
            **kwargs: 关键字参数。

        返回:
            调用父类的 p_mean_variance 方法的结果，使用包装后的模型。
        """
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        """
        计算训练损失。

        参数:
            model: 用于计算损失的模型。
            *args: 位置参数。
            **kwargs: 关键字参数。

        返回:
            调用父类的 training_losses 方法的结果，使用包装后的模型。
        """
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        """
        计算条件均值。

        参数:
            cond_fn: 条件函数。
            *args: 位置参数。
            **kwargs: 关键字参数。

        返回:
            调用父类的 condition_mean 方法的结果，使用包装后的条件函数。
        """
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        """
        计算条件分数。

        参数:
            cond_fn: 条件函数。
            *args: 位置参数。
            **kwargs: 关键字参数。

        返回:
            调用父类的 condition_score 方法的结果，使用包装后的条件函数。
        """
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        """
        包装模型以处理时间步映射和时间步缩放。

        参数:
            model: 要包装的模型。

        返回:
            _WrappedModel: 包装后的模型。
        """
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        """
        缩放时间步。在这个实现中，缩放由包装后的模型完成。

        参数:
            t: 要缩放的时间步。

        返回:
            未缩放的时间步。
        """
        # 缩放由包装后的模型完成
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        """
        初始化包装模型。

        参数:
            model: 要包装的原始模型。
            timestep_map (list): 时间步映射列表。
            rescale_timesteps (bool): 是否重新缩放时间步。
            original_num_steps (int): 原始的步数。
        """
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        """
        调用包装后的模型。

        参数:
            x: 输入数据。
            ts: 时间步。
            **kwargs: 其他关键字参数。

        返回:
            调用原始模型的结果，使用映射和可能缩放后的时间步。
        """
        # 将时间步映射转换为张量
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        # 获取新的时间步
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            # 重新缩放时间步
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)