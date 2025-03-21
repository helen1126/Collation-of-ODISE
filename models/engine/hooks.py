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

import inspect
import detectron2.utils.comm as comm
from detectron2.engine import EvalHook as _EvalHook
from detectron2.evaluation.testing import flatten_results_dict


class EvalHook(_EvalHook):
    def __init__(self, eval_period, eval_function):
        """
        初始化评估钩子。

        参数:
            eval_period (int): 评估的周期，即每隔多少个迭代进行一次评估。
            eval_function (callable): 评估函数，用于执行评估任务。该函数必须包含 'final_iter' 或 'next_iter' 作为参数。

        异常:
            AssertionError: 如果评估函数不包含 'final_iter' 或 'next_iter' 作为参数。
        """
        super().__init__(eval_period, eval_function)
        func_args = inspect.getfullargspec(eval_function).args
        assert {"final_iter", "next_iter"}.issubset(set(func_args)), (
            f"Eval function must have either 'final_iter' or 'next_iter' as an argument."
            f"Got {func_args} instead."
        )

    def _do_eval(self, final_iter=False, next_iter=0):
        """
        执行评估操作。

        参数:
            final_iter (bool, 可选): 是否为最后一次迭代，默认为 False。
            next_iter (int, 可选): 下一次迭代的编号，默认为 0。

        异常:
            AssertionError: 如果评估函数返回的结果不是字典类型。
            ValueError: 如果评估函数返回的嵌套字典中包含非浮点型的值。
        """
        results = self._func(final_iter=final_iter, next_iter=next_iter)

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_step(self):
        """
        在每个训练步骤之后执行的操作。
        如果达到评估周期且不是最后一次迭代，则执行评估操作。
        """
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._do_eval(next_iter=next_iter)

    def after_train(self):
        """
        在训练结束后执行的操作。
        如果训练正常结束，则执行最后一次评估操作，并清理评估函数以避免循环引用。
        """
        # This condition is to prevent the eval from running after a failed training
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval(final_iter=True)
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func