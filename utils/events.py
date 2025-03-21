# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import json
import logging
import os
import os.path as osp
from typing import Dict, Optional, Union
import torch
from detectron2.config import CfgNode
from detectron2.utils.events import CommonMetricPrinter as _CommonMetricPrinter
from detectron2.utils.events import EventWriter, get_event_storage


class WandbWriter(EventWriter):
    """
    Write all scalars to a wandb tool.
    based on https://github.com/facebookresearch/detectron2/pull/3716
    """

    def __init__(
        self,
        max_iter: int,
        run_name: str,
        output_dir: str,
        project: str = "ODISE",
        config: Union[Dict, CfgNode] = {},  # noqa: B006
        resume: bool = False,
        window_size: int = 20,
        **kwargs,
    ):
        """
        ��ʼ��WandbWriter�࣬���ڽ���������д��Wandb���ߡ�

        Args:
            max_iter (int): ������������
            run_name (str): ���е����ơ�
            output_dir (str): ���Ŀ¼��
            project (str, ��ѡ): W&B��Ŀ���ƣ�Ĭ��Ϊ "ODISE"��
            config (Union[Dict, CfgNode], ��ѡ): ��Ŀ��������ö���Ĭ��Ϊ���ֵ䡣
            resume (bool, ��ѡ): �Ƿ�ָ�֮ǰ�����У�Ĭ��ΪFalse��
            window_size (int, ��ѡ): ������ֵƽ���Ĵ��ڴ�С��Ĭ��Ϊ20��
            **kwargs: ���ݸ� `wandb.init(...)` ������������
        """
        logger = logging.getLogger(__name__)
        # ������Trainer.train()֮ǰ����ʧ��ʱ����
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        logger.info(f"Setting WANDB_START_METHOD to '{os.environ['WANDB_START_METHOD']}'")

        import wandb

        self._window_size = window_size
        self._run = (
            wandb.init(
                project=project,
                name=run_name,
                config=config,
                dir=output_dir,
                resume=resume,
                **kwargs,
            )
            if not wandb.run
            else wandb.run
        )
        self._run._label(repo="vision")
        self._max_iter = max_iter

        # �ֶ�д�� "wandb-resume.json" �ļ�
        # ����resume=Trueʱ��wandb.init()���Զ��������ļ�
        # ��˵�resume=Falseʱ�������ֶ����������Ա㼴ʹ֮ǰ������resume=False��Ҳ���Իָ�����
        resume_file = osp.join(output_dir, "wandb/wandb-resume.json")
        if not resume:
            logger.warning("Manually create wandb-resume.json file")
            with open(resume_file, "w") as f:
                json.dump({"run_id": self._run.id}, f)

    def write(self):
        """
        ���¼��洢�л�ȡ���µı������ݣ�������д��Wandb��
        ͬʱ���㵱ǰ���Ȳ���¼��
        """
        storage = get_event_storage()

        log_dict = {}
        for k, (v, _) in storage.latest_with_smoothing_hint(self._window_size).items():
            log_dict[k] = v

        log_dict["progress"] = storage.iter / self._max_iter

        self._run.log(log_dict)

    def close(self):
        """
        �ر�Wandb���С�
        ����ﵽ������������������������У����򣬱������δ��ɡ�
        """
        try:
            storage = get_event_storage()
            iteration = storage.iter
            if iteration >= self._max_iter - 1:
                # �ﵽ�������������������
                # finish()���Զ�ɾ�� "wandb-resume.json" �ļ�
                self._run.finish()
            else:
                # �������δ���
                self._run.finish(1)
        except AssertionError:
            # û��ѵ����/�¼��洢
            # �������δ���
            self._run.finish(1)


class CommonMetricPrinter(_CommonMetricPrinter):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.

    It's meant to print common metrics in common ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    """

    def __init__(self, max_iter: Optional[int] = None, window_size: int = 20, run_name: str = ""):
        """
        ��ʼ��CommonMetricPrinter�࣬���ڽ�����ָ���ӡ���նˡ�

        Args:
            max_iter (Optional[int], ��ѡ): ������������Ĭ��ΪNone��
            window_size (int, ��ѡ): ����ƽ��ָ��Ĵ��ڴ�С��Ĭ��Ϊ20��
            run_name (str, ��ѡ): ���е����ƣ�Ĭ��Ϊ���ַ�����
        """
        super().__init__(max_iter=max_iter, window_size=window_size)
        self.run_name = run_name

    def write(self):
        """
        ���¼��洢�л�ȡ����ָ�꣨�����ʱ�䡢ETA���ڴ桢��ʧ��ѧϰ�ʣ���
        �������ʽ�����ӡ���նˡ�
        ����ﵽ�������������򲻴�ӡ�κ����ݡ�
        """
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # ѵ���ɹ��󣬴˹��ӽ�����ѵ�����ȣ���ʧ��ETA�ȣ�����������������
            # ��˼�ʹ���ô˷�����Ҳ��д���κ�����
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # ��ǰ���ε����У�����������δʹ��SimpleTrainerʱ����Щָ����ܲ�����
            data_time = None
        try:
            iter_time = storage.history("time").global_avg()
        except KeyError:
            iter_time = None
        try:
            lr = "{:.5g}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        eta_string = self._get_eta(storage)

        storage_hide_keys = ["eta_seconds", "data_time", "time", "lr"]

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            "{run_name}  {eta}iter: {iter}{max_iter}  {losses}  {time}{data_time}lr: {lr}  {memory}".format(  # noqa: E501,B950
                run_name=self.run_name,
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                max_iter=f"/{self._max_iter}" if self._max_iter else "",
                # NOTE: Jiarui makes losses include all the storage.histories() except for
                # non-smoothing metrics. This hack will make writter log all the metrics in
                # the storage, but excluding metrics from EvalHook since smoothing_hint=False.
                losses="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(self._window_size))
                        for k, v in storage.histories().items()
                        if k not in storage_hide_keys and storage.smoothing_hints()[k]
                    ]
                ),
                time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
                data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )


class WriterStack:
    def __init__(self, logger, writers=None):
        """
        ��ʼ��WriterStack�࣬���ڹ����¼�д������

        Args:
            logger: ��־��¼����
            writers (list, ��ѡ): �¼�д�����б�Ĭ��ΪNone��
        """
        self.logger = logger
        self.writers = writers

    def __enter__(self):
        """
        ���������Ĺ�����ʱִ�еĲ��������ﲻ���κβ�����
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        �˳������Ĺ�����ʱִ�еĲ�����
        ��������쳣����¼������Ϣ���ر������¼�д������

        Args:
            exc_type: �쳣���͡�
            exc_val: �쳣ֵ��
            exc_tb: �쳣������Ϣ��
        """
        if exc_type is not None:
            self.logger.error("Error occurred in the writer", exc_info=(exc_type, exc_val, exc_tb))
            self.logger.error("Closing all writers")
            if self.writers is not None:
                for writer in self.writers:
                    writer.close()
            self.logger.error("All writers closed")