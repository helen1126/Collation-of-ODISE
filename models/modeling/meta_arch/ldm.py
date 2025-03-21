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
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from ldm.models.diffusion.ddpm import LatentDiffusion as _LatentDiffusion
from ldm.modules.diffusionmodules.openaimodel import timestep_embedding
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from timm.models.layers import trunc_normal_

from models.checkpoint.odise_checkpointer import LdmCheckpointer
from models.modeling.meta_arch.clip import ClipAdapter
from utils.file_io import PathManager

from ..diffusion import GaussianDiffusion, create_gaussian_diffusion
from ..preprocess import batched_input_to_device
from .helper import FeatureExtractor


def build_ldm_from_cfg(cfg_name) -> _LatentDiffusion:
    """
    根据配置文件名称构建潜在扩散模型（Latent Diffusion Model, LDM）。

    参数:
        cfg_name (str): 配置文件的名称。

    返回:
        _LatentDiffusion: 构建好的潜在扩散模型实例。
    """
    if cfg_name.startswith("v1"):
        url_prefix = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/"  # noqa
    elif cfg_name.startswith("v2"):
        url_prefix = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/"  # noqa

    logging.getLogger(__name__).info(f"Loading LDM config from {cfg_name}")
    config = OmegaConf.load(PathManager.open(url_prefix + cfg_name))
    return instantiate_from_config(config.model)


class DisableLogger:
    """
    禁用 Hugging Face 日志记录的上下文管理器。

    使用方法:
        with DisableLogger():
            # 在此代码块中，日志记录将被禁用
    """
    def __enter__(self):
        """
        进入上下文时，禁用关键级别的日志记录。
        """
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        """
        退出上下文时，恢复日志记录。

        参数:
            exit_type: 退出类型。
            exit_value: 退出值。
            exit_traceback: 退出时的回溯信息。
        """
        logging.disable(logging.NOTSET)


def add_device_property(cls):
    """
    为给定的类添加 `device` 属性，该属性返回模型的第一个参数所在的设备。

    参数:
        cls (class): 要添加属性的类。

    返回:
        class: 带有 `device` 属性的新类。
    """
    class TempClass(cls):
        pass

    TempClass.device = property(lambda m: next(m.parameters()).device)

    return TempClass


class LatentDiffusion(nn.Module):
    """
    LatentDiffusion 类用于实现基于潜在扩散模型的图像生成功能。
    该类集成了潜在扩散模型的各个组件，包括编码器、U-Net、解码器等，
    并提供了训练和推理的接口。
    """

    LDM_CONFIGS = {
        "sd://v1-3": ("v1-inference.yaml", (512, 512), (64, 64)),
        "sd://v1-4": ("v1-inference.yaml", (512, 512), (64, 64)),
        "sd://v1-5": ("v1-inference.yaml", (512, 512), (64, 64)),
        "sd://v2-0-base": ("v2-inference.yaml", (512, 512), (64, 64)),
        "sd://v2-0-v": ("v2-inference.yaml", (768, 768), (96, 96)),
        "sd://v2-1-base": ("v2-inference.yaml", (512, 512), (64, 64)),
        "sd://v2-1-v": ("v2-inference.yaml", (768, 768), (96, 96)),
    }

    def __init__(
        self,
        diffusion: Optional[GaussianDiffusion] = None,
        guidance_scale: float = 7.5,
        pixel_mean: Tuple[float] = (0.5, 0.5, 0.5),
        pixel_std: Tuple[float] = (0.5, 0.5, 0.5),
        init_checkpoint="sd://v1-3",
    ):
        """
        初始化 LatentDiffusion 类的实例。

        参数:
            diffusion (Optional[GaussianDiffusion]): 高斯扩散过程，默认为 None。
            guidance_scale (float): 引导尺度，用于控制文本引导的强度，默认为 7.5。
            pixel_mean (Tuple[float]): 图像像素的均值，用于归一化，默认为 (0.5, 0.5, 0.5)。
            pixel_std (Tuple[float]): 图像像素的标准差，用于归一化，默认为 (0.5, 0.5, 0.5)。
            init_checkpoint (str): 初始化检查点的名称，默认为 "sd://v1-3"。
        """
        super().__init__()
        self._logger = logging.getLogger(__name__)

        # 根据初始化检查点名称获取配置文件、图像大小和潜在图像大小
        ldm_cfg, image_size, latent_image_size = self.LDM_CONFIGS[init_checkpoint]

        # 禁用日志记录，加载潜在扩散模型
        with DisableLogger():
            self.ldm: _LatentDiffusion = build_ldm_from_cfg(ldm_cfg)
        # 为条件阶段模型添加设备属性
        self.ldm.cond_stage_model.__class__ = add_device_property(
            self.ldm.cond_stage_model.__class__
        )

        self.init_checkpoint = init_checkpoint
        # 加载预训练模型
        self.load_pretrain()

        self.image_size = image_size
        self.latent_image_size = latent_image_size

        self.latent_dim = self.ldm.channels
        assert self.latent_dim == self.ldm.first_stage_model.embed_dim
        # 如果未提供扩散过程，则创建一个默认的高斯扩散过程
        if diffusion is None:
            diffusion = create_gaussian_diffusion(
                steps=1000,
                learn_sigma=False,
                noise_schedule="ldm_linear",
                # timestep_respacing="ldm_ddim50",
            )
        self.diffusion = diffusion

        self.guidance_scale = guidance_scale

        # 注册无条件输入的缓冲区
        self.register_buffer("uncond_inputs", self.embed_text([""]))

        # 注册图像像素均值和标准差的缓冲区
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    def load_pretrain(self):
        """
        加载预训练模型的权重。
        使用 LdmCheckpointer 从指定的初始化检查点加载模型权重。
        """
        LdmCheckpointer(self.ldm).load(self.init_checkpoint)

    @property
    def device(self):
        """
        获取模型所在的设备。

        返回:
            torch.device: 模型所在的设备。
        """
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        前向传播函数，根据模型的训练或推理模式调用相应的方法。

        参数:
            batched_inputs (dict): 批量输入数据，包含图像和文本信息。

        返回:
            torch.Tensor: 生成的图像样本。
        """
        # 将批量输入数据移动到模型所在的设备上
        batched_inputs = batched_input_to_device(batched_inputs, next(self.parameters()).device)

        if self.training:
            return self.forward_train(batched_inputs)
        else:
            return self.forward_test(batched_inputs)

    def forward_train(self, batched_inputs):
        """
        训练模式下的前向传播函数。
        该方法目前未实现，调用时会抛出 NotImplementedError 异常。

        参数:
            batched_inputs (dict): 批量输入数据，包含图像和文本信息。

        抛出:
            NotImplementedError: 该方法未实现。
        """
        raise NotImplementedError

    def apply_model_with_guidence(self, x_noisy, t, cond):
        """
        应用模型并结合文本引导生成输出。

        参数:
            x_noisy (torch.Tensor): 带噪声的输入张量。
            t (torch.Tensor): 时间步张量。
            cond (torch.Tensor): 条件输入张量，通常是文本嵌入。

        返回:
            torch.Tensor: 应用模型并结合引导后的输出张量。
        """
        # 参考: https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # 取输入的前半部分
        half = x_noisy[: len(x_noisy) // 2]
        # 复制前半部分并拼接
        combined = torch.cat([half, half], dim=0)
        # 应用模型
        model_out = self.ldm.apply_model(combined, t, cond)
        # 分离噪声预测和其他部分
        eps, rest = model_out[:, : self.latent_dim], model_out[:, self.latent_dim :]
        # 分离条件噪声预测和无条件噪声预测
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # 结合引导计算半噪声预测
        half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
        # 拼接半噪声预测
        eps = torch.cat([half_eps, half_eps], dim=0)
        # 拼接噪声预测和其他部分
        return torch.cat([eps, rest], dim=1)

    def embed_text(self, text):
        """
        将文本转换为嵌入向量。

        参数:
            text (list): 输入的文本列表。

        返回:
            torch.Tensor: 文本的嵌入向量。
        """
        return self.ldm.get_learned_conditioning(text)

    @property
    def encoder(self):
        """
        获取编码器模块。

        返回:
            nn.Module: 编码器模块。
        """
        return self.ldm.first_stage_model.encoder

    @property
    def unet(self):
        """
        获取 U-Net 模块。

        返回:
            nn.Module: U-Net 模块。
        """
        return self.ldm.model.diffusion_model

    @property
    def decoder(self):
        """
        获取解码器模块。

        返回:
            nn.Module: 解码器模块。
        """
        return self.ldm.first_stage_model.decoder

    @torch.no_grad()
    def encode_to_latent(self, input_image):
        """
        将输入图像编码为潜在空间表示。

        参数:
            input_image (torch.Tensor): 输入的图像张量。

        返回:
            torch.Tensor: 编码后的潜在图像张量。
        """
        # 对输入图像进行编码
        encoder_posterior = self.ldm.encode_first_stage(input_image)
        # 为了使编码过程确定性，使用均值而不是从后验中采样
        latent_image = self.ldm.get_first_stage_encoding(encoder_posterior.mean)

        return latent_image

    @torch.no_grad()
    def decode_from_latent(self, latent_image):
        """
        将潜在空间表示解码为图像。

        参数:
            latent_image (torch.Tensor): 潜在图像张量。

        返回:
            torch.Tensor: 解码后的图像张量。
        """
        return self.ldm.decode_first_stage(latent_image)

    def forward_test(self, batched_inputs):
        """
        推理模式下的前向传播函数。

        参数:
            batched_inputs (dict): 批量输入数据，包含图像和文本信息。

        返回:
            torch.Tensor: 生成的图像样本。
        """
        # 获取输入的文本描述
        caption = batched_inputs["caption"]
        # 获取批量大小
        batch_size = len(caption)

        # 将文本描述转换为嵌入向量
        cond_inputs = self.embed_text(caption)

        # 如果引导尺度不为 1.0，则生成无条件输入
        if self.guidance_scale != 1.0:
            uncond_inputs = self.uncond_inputs.expand_as(cond_inputs)
        else:
            uncond_inputs = None

        # 如果无条件输入为空，则使用无引导的扩散采样
        if uncond_inputs is None:
            latent_samples = self.diffusion.ddim_sample_loop(
                model=self.ldm.apply_model,
                shape=(batch_size, self.latent_dim, *self.latent_image_size),
                device=self.device,
                clip_denoised=False,
                model_kwargs={"cond": cond_inputs},
            )
        else:
            # 否则，使用带引导的扩散采样
            latent_samples = self.diffusion.ddim_sample_loop(
                model=self.apply_model_with_guidence,
                shape=(batch_size * 2, self.latent_dim, *self.latent_image_size),
                device=self.device,
                clip_denoised=False,  # 对于 LDM 推理，不进行裁剪
                model_kwargs={"cond": torch.cat([cond_inputs, uncond_inputs], dim=0)},
            )[:batch_size]

        # 将潜在样本解码为图像
        decoded_samples = self.ldm.decode_first_stage(latent_samples)
        # 对解码后的图像进行反归一化
        out_samples = decoded_samples * self.pixel_std + self.pixel_mean
        # 将输出图像的像素值裁剪到 [0.0, 1.0] 范围内
        out_samples = out_samples.clamp(0.0, 1.0)

        return out_samples

class LdmExtractor(FeatureExtractor):
    def __init__(
        self,
        ldm: Optional[LatentDiffusion] = None,
        encoder_block_indices: Tuple[int, ...] = (5, 7),
        unet_block_indices: Tuple[int, ...] = (2, 5, 8, 11),
        decoder_block_indices: Tuple[int, ...] = (2, 5),
        steps: Tuple[int, ...] = (0,),
        share_noise: bool = True,
        enable_resize: bool = False,
    ):
        """
        初始化 LdmExtractor 类。

        参数:
            ldm (Optional[LatentDiffusion]): 潜在扩散模型实例，默认为 None。
            encoder_block_indices (Tuple[int, ...]): 编码器块的索引，用于提取特征，默认为 (5, 7)。
            unet_block_indices (Tuple[int, ...]): U-Net 块的索引，用于提取特征，默认为 (2, 5, 8, 11)。
            decoder_block_indices (Tuple[int, ...]): 解码器块的索引，用于提取特征，默认为 (2, 5)。
            steps (Tuple[int, ...]): 扩散步骤，默认为 (0,)。
            share_noise (bool): 是否共享噪声，默认为 True。
            enable_resize (bool): 是否启用图像缩放，默认为 False。
        """
        super().__init__()

        self.encoder_block_indices = encoder_block_indices
        self.unet_block_indices = unet_block_indices
        self.decoder_block_indices = decoder_block_indices

        self.steps = steps

        if ldm is not None:
            self.ldm = ldm
        else:
            self.ldm = LatentDiffusion()
        if enable_resize:
            self.image_preprocess = T.Resize(
                size=self.ldm.image_size, interpolation=T.InterpolationMode.BICUBIC
            )
        else:
            self.image_preprocess = None

        if share_noise:
            # 使用种子 42 生成共享噪声
            rng = torch.Generator().manual_seed(42)
            self.register_buffer(
                "shared_noise",
                torch.randn(1, self.ldm.latent_dim, *self.ldm.latent_image_size, generator=rng),
            )
        else:
            self.shared_noise = None

        self.reset_dim_stride()
        self._freeze()

    def reset_dim_stride(self):
        """
        重置特征维度和步长，并更新编码器、U-Net 和解码器的块列表。

        该函数会遍历编码器、U-Net 和解码器的各个块，根据指定的索引提取特征维度和步长，
        并将相应的块存储在类的属性中。

        返回:
            feature_dims (list): 特征维度列表。
            feature_strides (list): 特征步长列表。
        """
        # Encoder
        all_encoder_blocks = []
        for i_level in range(self.ldm.encoder.num_resolutions):
            for i_block in range(self.ldm.encoder.num_res_blocks):
                all_encoder_blocks.append(self.ldm.encoder.down[i_level].block[i_block])

        encoder_dims = []
        encoder_strides = []
        encoder_blocks = []
        for idx in self.encoder_block_indices:
            encoder_dims.append(all_encoder_blocks[idx].in_channels)
            group_size = 2
            encoder_strides.append(2 ** ((idx + group_size) // group_size - 1))
            encoder_blocks.append(all_encoder_blocks[idx])

        # UNet
        assert set(self.unet_block_indices).issubset(set(range(len(self.ldm.unet.output_blocks))))
        unet_dims = []
        unet_strides = []
        unet_blocks = []
        for idx, block in enumerate(self.ldm.unet.output_blocks):
            if idx in self.unet_block_indices:
                # The first block of TimestepEmbedSequential
                unet_dims.append(block[0].channels)

                group_size = 3
                unet_strides.append(64 // (2 ** ((idx + group_size) // group_size - 1)))
                unet_blocks.append(block)

        # Decoder
        all_decoder_blocks = []
        for i_level in reversed(range(self.ldm.decoder.num_resolutions)):
            for i_block in range(self.ldm.decoder.num_res_blocks + 1):
                all_decoder_blocks.append(self.ldm.decoder.up[i_level].block[i_block])

        decoder_dims = []
        decoder_strides = []
        decoder_blocks = []
        for idx in self.decoder_block_indices:
            decoder_dims.append(all_decoder_blocks[idx].in_channels)
            group_size = 3
            decoder_strides.append(8 // (2 ** ((idx + group_size) // group_size - 1)))
            decoder_blocks.append(all_decoder_blocks[idx])

        feature_dims = encoder_dims + unet_dims * len(self.steps) + decoder_dims
        feature_strides = encoder_strides + unet_strides * len(self.steps) + decoder_strides

        self.encoder_blocks = encoder_blocks
        self.unet_blocks = unet_blocks
        self.decoder_blocks = decoder_blocks

        return feature_dims, feature_strides

    @property
    def feature_size(self):
        """
        获取特征的尺寸。

        返回:
            tuple: 特征的尺寸，即潜在扩散模型的图像尺寸。
        """
        return self.ldm.image_size

    @property
    def feature_dims(self):
        """
        获取特征的维度。

        调用 reset_dim_stride 函数获取特征维度列表。

        返回:
            list: 特征维度列表。
        """
        return self.reset_dim_stride()[0]

    @property
    def feature_strides(self):
        """
        获取特征的步长。

        调用 reset_dim_stride 函数获取特征步长列表。

        返回:
            list: 特征步长列表。
        """
        return self.reset_dim_stride()[1]

    @property
    def num_groups(self) -> int:
        """
        获取特征组的数量。

        特征组的数量由编码器、U-Net 和解码器的块索引数量之和确定。

        返回:
            int: 特征组的数量。
        """
        num_groups = len(self.encoder_block_indices)
        num_groups += len(self.unet_block_indices)
        num_groups += len(self.decoder_block_indices)
        return num_groups

    @property
    def grouped_indices(self):
        """
        获取分组后的特征索引。

        该函数将编码器、U-Net 和解码器的特征索引进行分组，返回分组后的索引列表。

        返回:
            list: 分组后的特征索引列表。
        """
        ret = []

        for i in range(len(self.encoder_block_indices)):
            ret.append([i])

        offset = len(self.encoder_block_indices)

        for i in range(len(self.unet_block_indices)):
            cur_indices = []
            for t in range(len(self.steps)):
                cur_indices.append(i + t * len(self.unet_block_indices) + offset)
            ret.append(cur_indices)

        offset += len(self.steps) * len(self.unet_block_indices)

        for i in range(len(self.decoder_block_indices)):
            ret.append([i + offset])
        return ret

    @property
    def pixel_mean(self):
        """
        获取图像像素的均值。

        返回:
            torch.Tensor: 图像像素的均值。
        """
        return self.ldm.pixel_mean

    @property
    def pixel_std(self):
        """
        获取图像像素的标准差。

        返回:
            torch.Tensor: 图像像素的标准差。
        """
        return self.ldm.pixel_std

    @property
    def device(self):
        """
        获取模型所在的设备。

        返回:
            torch.device: 模型所在的设备。
        """
        return self.ldm.device

    @torch.no_grad()
    def build_text_embed(self, text: List[List[str]], batch_size=64, flatten=True):
        """
        构建文本嵌入。

        该函数将输入的文本列表转换为文本嵌入张量。

        参数:
            text (List[List[str]]): 输入的文本列表，每个子列表表示一个样本的文本。
            batch_size (int): 批量大小，默认为 64。
            flatten (bool): 是否将文本列表展平，默认为 True。

        返回:
            torch.Tensor: 文本嵌入张量。
        """
        if isinstance(text, str):
            text = [text]
        if isinstance(text[0], str):
            text = [[t] for t in text]

        # 检查输入是否为嵌套列表
        assert isinstance(text[0], list)

        # 展平嵌套列表
        flatten_text = [t for sublist in text for t in sublist]

        text_embed_list = []

        for i in range(0, len(flatten_text), batch_size):
            cur_text = flatten_text[i : i + batch_size]
            text_embed = self.ldm.embed_text(cur_text)
            text_embed_list.append(text_embed)

        return torch.concat(text_embed_list, dim=0)

    def encoder_forward(self, x):
        """
        编码器前向传播。

        该函数将输入的张量通过编码器进行前向传播，并提取指定块的特征。

        参数:
            x (torch.Tensor): 输入的张量。

        返回:
            h (torch.Tensor): 编码器的输出。
            ret_features (list): 提取的特征列表。
        """
        encoder = self.ldm.encoder
        ret_features = []

        # 时间步嵌入
        temb = None

        # 下采样
        hs = [encoder.conv_in(x)]
        for i_level in range(encoder.num_resolutions):
            for i_block in range(encoder.num_res_blocks):

                # 添加返回特征
                if encoder.down[i_level].block[i_block] in self.encoder_blocks:
                    ret_features.append(hs[-1].contiguous())

                h = encoder.down[i_level].block[i_block](hs[-1], temb)
                if len(encoder.down[i_level].attn) > 0:
                    h = encoder.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != encoder.num_resolutions - 1:
                hs.append(encoder.down[i_level].downsample(hs[-1]))

        # 中间层
        h = hs[-1]
        h = encoder.mid.block_1(h, temb)
        h = encoder.mid.attn_1(h)
        h = encoder.mid.block_2(h, temb)

        # 输出层
        h = encoder.norm_out(h)
        h = h * torch.sigmoid(h)
        h = encoder.conv_out(h)
        return h, ret_features

    def encode_to_latent(self, image: torch.Tensor):
        """
        将图像编码为潜在表示。

        该函数将输入的图像通过编码器进行前向传播，并将输出转换为潜在表示。

        参数:
            image (torch.Tensor): 输入的图像张量。

        返回:
            latent_image (torch.Tensor): 潜在表示张量。
            ret_features (list): 提取的特征列表。
        """
        h, ret_features = self.encoder_forward(image)
        moments = self.ldm.ldm.first_stage_model.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        # 为了使编码过程确定性，使用均值而不是从后验分布中采样
        latent_image = self.ldm.ldm.scale_factor * posterior.mean

        return latent_image, ret_features

    def unet_forward(self, x, timesteps, context, cond_emb=None):
        """
        U-Net 前向传播。

        该函数将输入的张量通过 U-Net 进行前向传播，并提取指定块的特征。

        参数:
            x (torch.Tensor): 输入的张量。
            timesteps (torch.Tensor): 时间步张量。
            context (torch.Tensor): 上下文张量。
            cond_emb (Optional[torch.Tensor]): 条件嵌入张量，默认为 None。

        返回:
            output (torch.Tensor): U-Net 的输出。
            ret_features (list): 提取的特征列表。
        """
        unet = self.ldm.unet
        ret_features = []

        hs = []
        t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False)
        emb = unet.time_embed(t_emb)
        if cond_emb is not None:
            emb += cond_emb

        h = x
        for module in unet.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = unet.middle_block(h, emb, context)
        for module in unet.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if module in self.unet_blocks:
                ret_features.append(h.contiguous())
            h = module(h, emb, context)
        return unet.out(h), ret_features

    def decoder_forward(self, z):
        """
        解码器前向传播。

        该函数将输入的潜在表示通过解码器进行前向传播，并提取指定块的特征。

        参数:
            z (torch.Tensor): 输入的潜在表示张量。

        返回:
            h (torch.Tensor): 解码器的输出。
            ret_features (list): 提取的特征列表。
        """
        decoder = self.ldm.decoder
        ret_features = []

        decoder.last_z_shape = z.shape

        # 时间步嵌入
        temb = None

        # 潜在表示到输入块
        h = decoder.conv_in(z)

        # 中间层
        h = decoder.mid.block_1(h, temb)
        h = decoder.mid.attn_1(h)
        h = decoder.mid.block_2(h, temb)

        # 上采样
        for i_level in reversed(range(decoder.num_resolutions)):
            for i_block in range(decoder.num_res_blocks + 1):

                # 添加返回特征
                if decoder.up[i_level].block[i_block] in self.decoder_blocks:
                    ret_features.append(h.contiguous())

                h = decoder.up[i_level].block[i_block](h, temb)
                if len(decoder.up[i_level].attn) > 0:
                    h = decoder.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = decoder.up[i_level].upsample(h)

        # 输出层
        if decoder.give_pre_end:
            return h

        h = decoder.norm_out(h)
        h = h * torch.sigmoid(h)
        h = decoder.conv_out(h)
        if decoder.tanh_out:
            h = torch.tanh(h)
        return h, ret_features

    def decode_to_image(self, z):
        """
        将潜在表示解码为图像。

        该函数将输入的潜在表示通过解码器进行前向传播，并将输出转换为图像。

        参数:
            z (torch.Tensor): 输入的潜在表示张量。

        返回:
            dec (torch.Tensor): 解码后的图像张量。
            ret_features (list): 提取的特征列表。
        """
        z = 1.0 / self.ldm.ldm.scale_factor * z

        z = self.ldm.ldm.first_stage_model.post_quant_conv(z)
        dec, ret_features = self.decoder_forward(z)

        return dec, ret_features

    def forward(self, batched_inputs):
        """
        前向传播函数。

        该函数将输入的批量数据进行前向传播，包括图像编码、U-Net 处理和解码过程，
        并返回提取的特征列表。

        参数:
            batched_inputs (dict): 批量输入数据，包含图像和可选的文本描述。

        返回:
            features (list): 提取的特征列表。
        """

        features = []

        image = batched_inputs["img"]
        batch_size = image.shape[0]

        if self.image_preprocess is None:
            normalized_image = (image - self.pixel_mean) / self.pixel_std
        else:
            normalized_image = self.image_preprocess((image - self.pixel_mean) / self.pixel_std)

        if "caption" in batched_inputs:
            captions = batched_inputs["caption"]
        else:
            captions = [""] * batch_size

        # latent_image = self.ldm.encode_to_latent(normalized_image)
        latent_image, encoder_features = self.encode_to_latent(normalized_image)
        cond_inputs = batched_inputs.get("cond_inputs", self.ldm.embed_text(captions))

        unet_features = []
        for i, t in enumerate(self.steps):

            if "cond_emb" in batched_inputs:
                cond_emb = batched_inputs["cond_emb"][:, i]
            else:
                cond_emb = None

            if t < 0:
                noisy_latent_image = latent_image
                # use 0 as no noise timestep
                t = torch.tensor([0], device=self.device).expand(batch_size)
            else:
                t = torch.tensor([t], device=self.device).expand(batch_size)
                if self.shared_noise is not None:
                    if self.shared_noise.shape[2:] != latent_image.shape[2:]:
                        assert self.image_preprocess is None
                        shared_noise = F.interpolate(
                            self.shared_noise,
                            size=latent_image.shape[2:],
                            mode="bicubic",
                            align_corners=False,
                        )
                    else:
                        shared_noise = self.shared_noise
                    noise = shared_noise.expand_as(latent_image)
                else:
                    noise = None

                noisy_latent_image = self.ldm.diffusion.q_sample(latent_image, t, noise)
            # self.ldm.ldm.apply_model(noisy_latent_image, t, cond_inputs)
            _, cond_unet_features = self.unet_forward(
                noisy_latent_image, t, cond_inputs, cond_emb=cond_emb
            )
            unet_features.extend(cond_unet_features)

        # self.ldm.decode_from_latent(latent_image)
        _, decoder_features = self.decode_to_image(latent_image)

        features = [*encoder_features, *unet_features, *decoder_features]

        assert len(features) == len(
            self.feature_dims
        ), f"{len(features)} != {len(self.feature_dims)}"

        for indices in self.grouped_indices:
            for idx in indices:
                if self.image_preprocess is not None:
                    continue
                assert image.shape[-2] // self.feature_strides[idx] == features[idx].shape[-2]
                assert image.shape[-1] // self.feature_strides[idx] == features[idx].shape[-1]

        return features


class PositionalLinear(nn.Module):
    def __init__(self, in_features, out_features, seq_len=77, bias=True):
        """
        初始化 PositionalLinear 类。

        参数:
            in_features (int): 输入特征的维度。
            out_features (int): 输出特征的维度。
            seq_len (int, 可选): 序列的长度，默认为 77。
            bias (bool, 可选): 是否使用偏置，默认为 True。
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, out_features))
        trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入的张量。

        返回:
            torch.Tensor: 经过线性变换和位置嵌入后的输出张量。
        """
        x = self.linear(x)
        x = x.unsqueeze(1) + self.positional_embedding

        return x


class LdmImplicitCaptionerExtractor(nn.Module):
    def __init__(
        self,
        learnable_time_embed=True,
        num_timesteps=1,
        clip_model_name="ViT-L-14",
        **kwargs,
    ):
        """
        初始化 LdmImplicitCaptionerExtractor 类。

        参数:
            learnable_time_embed (bool, 可选): 是否使用可学习的时间嵌入，默认为 True。
            num_timesteps (int, 可选): 时间步的数量，默认为 1。
            clip_model_name (str, 可选): CLIP 模型的名称，默认为 "ViT-L-14"。
            **kwargs: 传递给 LdmExtractor 类的其他参数。
        """
        super().__init__()

        self.ldm_extractor = LdmExtractor(**kwargs)

        self.text_embed_shape = self.ldm_extractor.ldm.embed_text([""]).shape[1:]

        self.clip = ClipAdapter(name=clip_model_name, normalize=False)

        self.clip_project = PositionalLinear(
            self.clip.dim_latent, self.text_embed_shape[1], self.text_embed_shape[0]
        )
        self.alpha_cond = nn.Parameter(torch.zeros_like(self.ldm_extractor.ldm.uncond_inputs))

        self.learnable_time_embed = learnable_time_embed

        if self.learnable_time_embed:
            # self.ldm_extractor.ldm.unet.time_embed is nn.Sequential
            self.time_embed_project = PositionalLinear(
                self.clip.dim_latent,
                self.ldm_extractor.ldm.unet.time_embed[-1].out_features,
                num_timesteps,
            )
            self.alpha_cond_time_embed = nn.Parameter(
                torch.zeros(self.ldm_extractor.ldm.unet.time_embed[-1].out_features)
            )

    @property
    def feature_size(self):
        """
        获取特征图的尺寸。

        返回:
            tuple: 特征图的尺寸。
        """
        return self.ldm_extractor.feature_size

    @property
    def feature_dims(self):
        """
        获取特征图的维度。

        返回:
            list: 特征图的维度列表。
        """
        return self.ldm_extractor.feature_dims

    @property
    def feature_strides(self):
        """
        获取特征图的步长。

        返回:
            list: 特征图的步长列表。
        """
        return self.ldm_extractor.feature_strides

    @property
    def num_groups(self) -> int:
        """
        获取特征组的数量。

        返回:
            int: 特征组的数量。
        """
        return self.ldm_extractor.num_groups

    @property
    def grouped_indices(self):
        """
        获取分组后的索引。

        返回:
            list: 分组后的索引列表。
        """
        return self.ldm_extractor.grouped_indices

    def extra_repr(self):
        """
        返回额外的表示信息。

        返回:
            str: 额外的表示信息，包含 learnable_time_embed 的值。
        """
        return f"learnable_time_embed={self.learnable_time_embed}"

    def forward(self, batched_inputs):
        """
        前向传播函数。

        参数:
            batched_inputs (dict): 输入的批量数据，期望的键为 "img"（图像数据）和可选的 "caption"（文本描述）。

        返回:
            list: 提取的特征列表。
        """
        image = batched_inputs["img"]

        prefix = self.clip.embed_image(image).image_embed
        prefix_embed = self.clip_project(prefix)
        batched_inputs["cond_inputs"] = (
            self.ldm_extractor.ldm.uncond_inputs + torch.tanh(self.alpha_cond) * prefix_embed
        )

        if self.learnable_time_embed:
            batched_inputs["cond_emb"] = torch.tanh(
                self.alpha_cond_time_embed
            ) * self.time_embed_project(prefix)

        self.set_requires_grad(self.training)

        return self.ldm_extractor(batched_inputs)

    def set_requires_grad(self, requires_grad):
        """
        设置模型参数的可训练性。

        参数:
            requires_grad (bool): 是否允许参数进行梯度更新。
        """
        for p in self.ldm_extractor.ldm.ldm.model.parameters():
            p.requires_grad = requires_grad