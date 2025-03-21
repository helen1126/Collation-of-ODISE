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
    ��ԭʼ����ɢ�����д���һ��Ҫʹ�õ�ʱ�䲽�б���������ϣ����ԭʼ���̵ĵȴ�С������ѡȡ��ʱ�䲽����ȷ����

    ���磬���ԭʼ��ɢ������300��ʱ�䲽���Ҳ��ּ���Ϊ [10, 15, 20]��
    ��ôǰ100��ʱ�䲽����������Ϊ10��ʱ�䲽���ڶ���100��ʱ�䲽����������Ϊ15��ʱ�䲽��
    ���100��ʱ�䲽����������Ϊ20��ʱ�䲽��

    ���������һ���� "ddim" ��ͷ���ַ�������ʹ��DDIM�����еĹ̶�����������ֻ����һ�����֡�

    ����:
        num_timesteps (int): ԭʼ��ɢ������Ҫ���ֵ���ɢ������
        section_counts (list or str): һ�������б�����һ���������ŷָ����ֵ��ַ�����
                                      ��ʾÿ�����ֵĲ�������������£�ʹ�� "ddimN"������N��
                                      Ҫʹ�õ�DDIM�����еĲ�����

    ����:
        set: һ������ԭʼ��ɢ���̵�ʱ�䲽���ϣ����ں�������ɢ���̡�
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            # ��ȡ�����Ĳ���
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"�޷�ʹ������������ȷ���� {num_timesteps} ������")
        elif section_counts.startswith("ldm_ddim"):
            # ��ȡ�����Ĳ���
            desired_count = int(section_counts[len("ldm_ddim") :])
            # ��ddim��ȣ���1��ȷ�������ڼ����յ�alphaֵ��ȷ
            # �ο�: https://github.com/CompVis/stable-diffusion/blob/d39f5b51a8d607fd855425a0d546b9f871034c3d/ldm/modules/diffusionmodules/util.py#L56
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(1, num_timesteps + 1, i))
            raise ValueError(f"�޷�ʹ������������ȷ���� {num_timesteps} ������")
        elif section_counts == "fast27":
            # �ݹ���� space_timesteps ����
            steps = space_timesteps(num_timesteps, "10,10,3,2,2")
            # ��������������ʱ�䲽��DDIMαӰ
            steps.remove(num_timesteps - 1)
            steps.add(num_timesteps - 3)
            return steps
        # ���ַ���ת��Ϊ�����б�
        section_counts = [int(x) for x in section_counts.split(",")]
    # ����ÿ�����ֵĴ�С
    size_per = num_timesteps // len(section_counts)
    # ����ʣ��Ĳ���
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        # ���㵱ǰ���ֵ�ʵ�ʴ�С
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"�޷��� {size} ������Ĳ��ֻ���Ϊ {section_count} ������")
        if section_count <= 1:
            frac_stride = 1
        else:
            # �����������
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
    һ�������ڻ�����ɢ�����������������ɢ���̡�

    ����:
        use_timesteps (collection): һ������ԭʼ��ɢ���̵�ʱ�䲽���ϣ����л򼯺ϣ�����ʾҪ������ʱ�䲽��
        **kwargs: ���ڴ���������ɢ���̵Ĺؼ��ֲ�����
    """

    def __init__(self, use_timesteps, **kwargs):
        # ��ʹ�õ�ʱ�䲽ת��Ϊ����
        self.use_timesteps = set(use_timesteps)
        # �洢ʱ�䲽ӳ��
        self.timestep_map = []
        # �洢ԭʼ�Ĳ���
        self.original_num_steps = len(kwargs["betas"])

        # ����������ɢ����
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                # �����µ�betaֵ
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                # ��¼ʱ�䲽ӳ��
                self.timestep_map.append(i)
        # ����betas����
        kwargs["betas"] = np.array(new_betas)
        # ���ø���Ĺ��캯��
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        """
        �������ģ���µľ�ֵ�ͷ��

        ����:
            model: ���ڼ����ģ�͡�
            *args: λ�ò�����
            **kwargs: �ؼ��ֲ�����

        ����:
            ���ø���� p_mean_variance �����Ľ����ʹ�ð�װ���ģ�͡�
        """
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        """
        ����ѵ����ʧ��

        ����:
            model: ���ڼ�����ʧ��ģ�͡�
            *args: λ�ò�����
            **kwargs: �ؼ��ֲ�����

        ����:
            ���ø���� training_losses �����Ľ����ʹ�ð�װ���ģ�͡�
        """
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        """
        ����������ֵ��

        ����:
            cond_fn: ����������
            *args: λ�ò�����
            **kwargs: �ؼ��ֲ�����

        ����:
            ���ø���� condition_mean �����Ľ����ʹ�ð�װ�������������
        """
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        """
        ��������������

        ����:
            cond_fn: ����������
            *args: λ�ò�����
            **kwargs: �ؼ��ֲ�����

        ����:
            ���ø���� condition_score �����Ľ����ʹ�ð�װ�������������
        """
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        """
        ��װģ���Դ���ʱ�䲽ӳ���ʱ�䲽���š�

        ����:
            model: Ҫ��װ��ģ�͡�

        ����:
            _WrappedModel: ��װ���ģ�͡�
        """
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        """
        ����ʱ�䲽�������ʵ���У������ɰ�װ���ģ����ɡ�

        ����:
            t: Ҫ���ŵ�ʱ�䲽��

        ����:
            δ���ŵ�ʱ�䲽��
        """
        # �����ɰ�װ���ģ�����
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        """
        ��ʼ����װģ�͡�

        ����:
            model: Ҫ��װ��ԭʼģ�͡�
            timestep_map (list): ʱ�䲽ӳ���б�
            rescale_timesteps (bool): �Ƿ���������ʱ�䲽��
            original_num_steps (int): ԭʼ�Ĳ�����
        """
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        """
        ���ð�װ���ģ�͡�

        ����:
            x: �������ݡ�
            ts: ʱ�䲽��
            **kwargs: �����ؼ��ֲ�����

        ����:
            ����ԭʼģ�͵Ľ����ʹ��ӳ��Ϳ������ź��ʱ�䲽��
        """
        # ��ʱ�䲽ӳ��ת��Ϊ����
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        # ��ȡ�µ�ʱ�䲽
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            # ��������ʱ�䲽
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)