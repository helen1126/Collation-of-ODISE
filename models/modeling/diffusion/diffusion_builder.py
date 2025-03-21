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

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    """
    创建一个高斯扩散模型实例。

    参数:
        steps (int, 可选): 扩散过程的总步数。默认为 1000。
        learn_sigma (bool, 可选): 是否学习噪声标准差。默认为 False。
        sigma_small (bool, 可选): 是否使用小的固定噪声标准差。默认为 False。
        noise_schedule (str, 可选): 噪声调度类型，例如 "linear"。默认为 "linear"。
        use_kl (bool, 可选): 是否使用 KL 散度作为损失类型。默认为 False。
        predict_xstart (bool, 可选): 模型是否预测起始状态 x_0。默认为 False。
        rescale_timesteps (bool, 可选): 是否重新缩放时间步长。默认为 False。
        rescale_learned_sigmas (bool, 可选): 是否重新缩放学习到的噪声标准差。默认为 False。
        timestep_respacing (str, 可选): 时间步长间隔策略，例如 "20" 表示每隔 20 步采样。默认为 ""。

    返回:
        SpacedDiffusion: 一个高斯扩散模型实例，用于生成图像或其他数据。
    """
    # 根据指定的噪声调度类型和步数，获取 beta 序列
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    # 根据参数确定损失类型
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    # 如果没有指定时间步长间隔策略，则默认使用所有步数
    if not timestep_respacing:
        timestep_respacing = [steps]
    # 创建并返回 SpacedDiffusion 实例
    return SpacedDiffusion(
        # 根据步数和时间步长间隔策略，确定要使用的时间步长
        use_timesteps=space_timesteps(steps, timestep_respacing),
        # 传入计算得到的 beta 序列
        betas=betas,
        # 根据 predict_xstart 参数确定模型预测的均值类型
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        # 根据 learn_sigma 和 sigma_small 参数确定模型的方差类型
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        # 传入确定好的损失类型
        loss_type=loss_type,
        # 传入是否重新缩放时间步长的参数
        rescale_timesteps=rescale_timesteps,
    )