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
    ����һ����˹��ɢģ��ʵ����

    ����:
        steps (int, ��ѡ): ��ɢ���̵��ܲ�����Ĭ��Ϊ 1000��
        learn_sigma (bool, ��ѡ): �Ƿ�ѧϰ������׼�Ĭ��Ϊ False��
        sigma_small (bool, ��ѡ): �Ƿ�ʹ��С�Ĺ̶�������׼�Ĭ��Ϊ False��
        noise_schedule (str, ��ѡ): �����������ͣ����� "linear"��Ĭ��Ϊ "linear"��
        use_kl (bool, ��ѡ): �Ƿ�ʹ�� KL ɢ����Ϊ��ʧ���͡�Ĭ��Ϊ False��
        predict_xstart (bool, ��ѡ): ģ���Ƿ�Ԥ����ʼ״̬ x_0��Ĭ��Ϊ False��
        rescale_timesteps (bool, ��ѡ): �Ƿ���������ʱ�䲽����Ĭ��Ϊ False��
        rescale_learned_sigmas (bool, ��ѡ): �Ƿ���������ѧϰ����������׼�Ĭ��Ϊ False��
        timestep_respacing (str, ��ѡ): ʱ�䲽��������ԣ����� "20" ��ʾÿ�� 20 ��������Ĭ��Ϊ ""��

    ����:
        SpacedDiffusion: һ����˹��ɢģ��ʵ������������ͼ����������ݡ�
    """
    # ����ָ���������������ͺͲ�������ȡ beta ����
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    # ���ݲ���ȷ����ʧ����
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    # ���û��ָ��ʱ�䲽��������ԣ���Ĭ��ʹ�����в���
    if not timestep_respacing:
        timestep_respacing = [steps]
    # ���������� SpacedDiffusion ʵ��
    return SpacedDiffusion(
        # ���ݲ�����ʱ�䲽��������ԣ�ȷ��Ҫʹ�õ�ʱ�䲽��
        use_timesteps=space_timesteps(steps, timestep_respacing),
        # �������õ��� beta ����
        betas=betas,
        # ���� predict_xstart ����ȷ��ģ��Ԥ��ľ�ֵ����
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        # ���� learn_sigma �� sigma_small ����ȷ��ģ�͵ķ�������
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        # ����ȷ���õ���ʧ����
        loss_type=loss_type,
        # �����Ƿ���������ʱ�䲽���Ĳ���
        rescale_timesteps=rescale_timesteps,
    )