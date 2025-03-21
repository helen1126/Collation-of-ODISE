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
        初始化WandbWriter类，用于将标量数据写入Wandb工具。

        Args:
            max_iter (int): 最大迭代次数。
            run_name (str): 运行的名称。
            output_dir (str): 输出目录。
            project (str, 可选): W&B项目名称，默认为 "ODISE"。
            config (Union[Dict, CfgNode], 可选): 项目级别的配置对象，默认为空字典。
            resume (bool, 可选): 是否恢复之前的运行，默认为False。
            window_size (int, 可选): 用于中值平滑的窗口大小，默认为20。
            **kwargs: 传递给 `wandb.init(...)` 的其他参数。
        """
        logger = logging.getLogger(__name__)
        # 避免在Trainer.train()之前进程失败时挂起
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

        # 手动写入 "wandb-resume.json" 文件
        # 仅当resume=True时，wandb.init()会自动创建该文件
        # 因此当resume=False时，我们手动创建它，以便即使之前传递了resume=False，也可以恢复运行
        resume_file = osp.join(output_dir, "wandb/wandb-resume.json")
        if not resume:
            logger.warning("Manually create wandb-resume.json file")
            with open(resume_file, "w") as f:
                json.dump({"run_id": self._run.id}, f)

    def write(self):
        """
        从事件存储中获取最新的标量数据，并将其写入Wandb。
        同时计算当前进度并记录。
        """
        storage = get_event_storage()

        log_dict = {}
        for k, (v, _) in storage.latest_with_smoothing_hint(self._window_size).items():
            log_dict[k] = v

        log_dict["progress"] = storage.iter / self._max_iter

        self._run.log(log_dict)

    def close(self):
        """
        关闭Wandb运行。
        如果达到最大迭代次数，则正常完成运行；否则，标记运行未完成。
        """
        try:
            storage = get_event_storage()
            iteration = storage.iter
            if iteration >= self._max_iter - 1:
                # 达到最大迭代次数后完成运行
                # finish()会自动删除 "wandb-resume.json" 文件
                self._run.finish()
            else:
                # 标记运行未完成
                self._run.finish(1)
        except AssertionError:
            # 没有训练器/事件存储
            # 标记运行未完成
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
        初始化CommonMetricPrinter类，用于将常见指标打印到终端。

        Args:
            max_iter (Optional[int], 可选): 最大迭代次数，默认为None。
            window_size (int, 可选): 用于平滑指标的窗口大小，默认为20。
            run_name (str, 可选): 运行的名称，默认为空字符串。
        """
        super().__init__(max_iter=max_iter, window_size=window_size)
        self.run_name = run_name

    def write(self):
        """
        从事件存储中获取常见指标（如迭代时间、ETA、内存、损失和学习率），
        并将其格式化后打印到终端。
        如果达到最大迭代次数，则不打印任何内容。
        """
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # 训练成功后，此钩子仅报告训练进度（损失、ETA等），不报告其他数据
            # 因此即使调用此方法，也不写入任何内容
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # 在前几次迭代中（由于热身）或未使用SimpleTrainer时，这些指标可能不存在
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
        初始化WriterStack类，用于管理事件写入器。

        Args:
            logger: 日志记录器。
            writers (list, 可选): 事件写入器列表，默认为None。
        """
        self.logger = logger
        self.writers = writers

    def __enter__(self):
        """
        进入上下文管理器时执行的操作，这里不做任何操作。
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文管理器时执行的操作。
        如果发生异常，记录错误信息并关闭所有事件写入器。

        Args:
            exc_type: 异常类型。
            exc_val: 异常值。
            exc_tb: 异常回溯信息。
        """
        if exc_type is not None:
            self.logger.error("Error occurred in the writer", exc_info=(exc_type, exc_val, exc_tb))
            self.logger.error("Closing all writers")
            if self.writers is not None:
                for writer in self.writers:
                    writer.close()
            self.logger.error("All writers closed")