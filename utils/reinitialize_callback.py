# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import distributed as dist
from transformers import TrainerCallback

from models.utils_quant import QuantizeLinear


class ReinitializeCallback(TrainerCallback):
    """Periodically interpolates model weights with their quantized version."""

    def __init__(self, training_args):
        self.training_args = training_args

    def on_step_end(self, args, state, control, **kwargs):
        if not self.training_args.reinitialize_weights:
            return

        step = state.global_step
        if step == 0 or step % self.training_args.reinitialize_steps != 0:
            return

        model = kwargs.get("model")
        if model is None:
            return

        self._interpolate_weights(model)

    def _interpolate_weights(self, model):
        alpha = float(self.training_args.reinitialize_alpha)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        with torch.no_grad():
            for module in model.modules():
                if not isinstance(module, QuantizeLinear):
                    continue

                quantized_weight = module.quantized_weight(
                    dtype=module.weight.dtype, apply_noise=False
                )
                module.weight.data.mul_(alpha).add_(quantized_weight * (1 - alpha))
                module.reset_noise_buffers()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
