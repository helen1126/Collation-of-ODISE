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
    ����ɵ����������ݶȷ�����
    �÷����ǽ������ݶ���Ϊһ����һ����������ġ�

    ����:
        parameters (Iterable[Tensor] or Tensor): һ���ɵ����������򵥸����������ݶȽ�����һ����
        norm_type (float or int): ��ʹ�õ� p �������͡������� 'inf' ��ʾ�������

    ����:
        torch.Tensor: ���������ݶȷ�������Ϊһ����һ��������
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
        ��ʼ�� SimpleTrainer �ࡣ

        ����:
            model: Ҫѵ����ģ�͡�
            data_loader: ���ݼ������������ṩѵ�����ݡ�
            optimizer: �Ż��������ڸ���ģ�Ͳ�����
            grad_clip (��ѡ): �ݶȲü�����ֵ��
        """
        super().__init__(model, data_loader, optimizer)
        self.grad_clip = grad_clip
        logger.info(f"Trainer: {self.__class__.__name__}")
        logger.info(f"grad_clip: {grad_clip}")
        logger.info("All parameters: \n" + parameter_count_table(model))

        # ��ӡ��ѵ������
        logger.info("Trainable parameters: \n" + parameter_count_table(model, trainable_only=True))

    def raise_loss_nan(self, losses):
        """
        �����ʧ�Ƿ�Ϊ NaN ���������������׳��쳣��

        ����:
            losses (torch.Tensor): ��ʧ������
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
        ʵ�ֱ�׼��ѵ���߼���
        �˷���ִ�����²��裺
        1. �����ݼ������л�ȡ���ݡ�
        2. ǰ�򴫲�������ʧ��
        3. ���򴫲������ݶȡ�
        4. ��ѡ�ؽ����ݶȲü���
        5. ��¼�ݶȷ�����
        6. ��¼��ʧָ�ꡣ
        7. ����ģ�Ͳ�����
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        �����Ҫ�����ݽ��д������԰�װ���ݼ�������
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        �����Ҫ����ʧ���д������԰�װģ�͡�
        """
        # ע�⣺����������ֵ䣬��� "runner_meta"���� Jiarui ���
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
        �����Ҫ�ۻ��ݶȻ�������Ʋ��������԰�װ�Ż����� zero_grad() ������
        """
        self.optimizer.zero_grad()
        losses.backward()

        # ע�⣺��� "grad_norm"���� Jiarui ���
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        else:
            grad_norm = get_grad_norm(self.model.parameters())
        self.storage.put_scalar("grad_norm", grad_norm)

        # ���ô˼�飬�鿴�Ƿ��Ҫ
        # �ڼ�¼ָ��֮ǰ��� NaN���˳����н���
        # self.raise_loss_nan(losses)

        self._write_metrics(loss_dict, data_time)

        """
        �����Ҫ�ݶȲü�/���Ż������������԰�װ�Ż����� step() ������
        ���� https://arxiv.org/abs/2006.15704 Sec 3.2.4 ���������ַ����������ŵġ�
        """
        self.optimizer.step()

    # �븸�༸����ͬ�����˼�¼���� `metric_dict` ������
    # ԭʼ detectron2 �е� len(metric_dict) > 1�������Ǹ� bug
    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        """
        ��¼��ʧָ�ꡣ

        ����:
            loss_dict (dict): ������ʧ���ֵ䡣
            data_time (float): ���ݼ��������������ѵ�ʱ�䡣
            prefix (str, ��ѡ): ��־����ǰ׺��
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # �����й����������ռ�ָ���Խ�����־��¼
        # ���������ʹ�� DDP ����ѵ��������Ŀǰ detectron2 Ψһ֧�ֵķ�����
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # ��������֮������ݼ���ʱ������кܴ���졣
            # ʵ�������ݼ���ʱ��������ӳ������й��������е����ֵ��
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # ������ָ����ƽ��ֵ
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
    �ο�: https://github.com/microsoft/Swin-Transformer/blob/afeb877fba1139dfbc186276983af2abb02c2196/main.py#L194
    """  # noqa

    state_dict_key = "amp_scaler"

    def __init__(self):
        """
        ��ʼ�� NativeScalerWithGradNormCount �ࡣ
        ����һ�� PyTorch �� GradScaler �����Զ������ݶȡ�
        """
        from torch.cuda.amp import GradScaler

        self._scaler = GradScaler()

    def __call__(
        self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True
    ):
        """
        ִ���ݶ����š����򴫲����ݶȲü����Ż������²�����

        ����:
            loss (torch.Tensor): ��ʧ������
            optimizer: �Ż�����
            clip_grad (��ѡ): �ݶȲü�����ֵ��
            parameters (��ѡ): Ҫ�����ݶȷ����Ĳ�����
            create_graph (bool, ��ѡ): �Ƿ񴴽�����ͼ�����ڶ��׵������㡣
            update_grad (bool, ��ѡ): �Ƿ�����ݶȡ�

        ����:
            torch.Tensor: �ݶȷ�����
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # �͵�ȡ���Ż����������ݶ�����
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
        ��ȡ GradScaler ��״̬�ֵ䡣

        ����:
            dict: GradScaler ��״̬�ֵ䡣
        """
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """
        ���� GradScaler ��״̬�ֵ䡣

        ����:
            state_dict (dict): GradScaler ��״̬�ֵ䡣
        """
        self._scaler.load_state_dict(state_dict)


def get_optimizer_parameters(optimizer):
    """
    ��ȡ�Ż����е����в�����

    ����:
        optimizer: �Ż�����

    ����:
        itertools.chain: �Ż����е����в����ĵ�������
    """
    return itertools.chain(*[x["params"] for x in optimizer.param_groups])


class AMPTrainer(SimpleTrainer):
    """
    ������ `SimpleTrainer`������ѵ��ѭ����ʹ�� PyTorch ��ԭ���Զ���Ͼ��ȣ�AMP����
    """

    def __init__(self, model, data_loader, optimizer, grad_clip=None):
        """
        ��ʼ�� AMPTrainer �ࡣ

        ����:
            model: Ҫѵ����ģ�͡�
            data_loader: ���ݼ������������ṩѵ�����ݡ�
            optimizer: �Ż��������ڸ���ģ�Ͳ�����
            grad_clip (��ѡ): �ݶȲü�����ֵ��
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        # ע�⣺Jiarui �ſ��˴˼�飬��Ϊ���� AMP ��˵���Ǳ����
        elif isinstance(model, DataParallel):
            assert not len(model.device_ids) > 1, unsupported
        # assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer, grad_clip=grad_clip)

        self.grad_scaler = NativeScalerWithGradNormCount()

    def run_step(self):
        """
        ʵ�� AMP ѵ���߼���
        �˷���ִ�����²��裺
        1. �����ݼ������л�ȡ���ݡ�
        2. ʹ���Զ���Ͼ��Ƚ���ǰ�򴫲�������ʧ��
        3. ����ģ�Ͳ������ݶȡ�
        4. ʹ�� GradScaler �����ݶ����š����򴫲����ݶȲü����Ż������¡�
        5. ��¼�ݶȷ�������ʧ����ֵ��
        6. ��¼��ʧָ�ꡣ
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # detectron2 ʹ�ü򵥵� collate ��������������ֻ��һ���ֵ��б�
        # ���ǲ�Ϊ�������Ԫ��Ϣ
        # ���� webdataset��������һ�������ֵ�
        if isinstance(data, dict):
            # ע�⣺��� "runner_meta"���� Jiarui ���
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

        # ����ʹ�� self.optimizer.zero_grad()��
        # ���ǽ�ģ�������в������ݶ�����
        self.model.zero_grad()

        # ע�⣺���²�����װ�� NativeScalerWithGradNormCount() �� __call__() �У�
        # self.grad_scaler.scale(losses).backward()
        # self.grad_scaler.step(self.optimizer)
        # self.grad_scaler.update()
        grad_norm = self.grad_scaler(
            losses,
            self.optimizer,
            clip_grad=self.grad_clip,
            # ����ʹ�� self.model.parameters()������ʹ���Ż����еĲ���
            # �������ǾͲ���ü������Ż����еĲ���
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
        ��ȡѵ������״̬�ֵ䣬���� GradScaler ��״̬��

        ����:
            dict: ѵ������״̬�ֵ䡣
        """
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        """
        ����ѵ������״̬�ֵ䣬���� GradScaler ��״̬��

        ����:
            state_dict (dict): ѵ������״̬�ֵ䡣
        """
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])