# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math

import torch
import transformers
from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant
from torch import distributed as dist
from transformers import default_data_collator, Trainer
from utils import datautils, utils

from utils.process_args import process_args
from utils.reinitialize_callback import ReinitializeCallback

log = utils.get_logger("clm")


def train():
    dist.init_process_group(backend="nccl")
    model_args, data_args, training_args = process_args()

    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float

    config = LlamaConfig.from_pretrained(model_args.input_model_filename)
    config.w_bits = model_args.w_bits
    config.share_embedding = model_args.share_embedding
    if model_args.share_embedding:
        config.tie_word_embeddings = True
    if model_args.wrap_layer_xl:
        config.wrap_layer_xl = True
    # Configure noise injection parameters (WinQ)
    noise_enabled = (
        training_args.noise_injection
        or training_args.nso_within_model
        or training_args.train_nso
    )
    config.noise_injection = noise_enabled
    config.pre_quantization_noise = (
        training_args.pre_quantization_noise or training_args.nso_pre_quantization_noise
    )
    config.post_quantization_noise = (
        training_args.post_quantization_noise
        or training_args.nso_post_quantization_noise
    )
    config.initialize_noise = (
        training_args.initialize_noise or training_args.nso_initialize_noise
    )
    config.trainable_noise_scale = (
        training_args.trainable_noise_scale or training_args.nso_trainable_noise_scale
    )
    config.noise_sigma_weights = (
        training_args.noise_sigma_weights
        if training_args.noise_sigma_weights != 0.0
        else training_args.nso_sigma_weights
    )
    config.noise_sigma_clipvals = (
        training_args.noise_sigma_clipvals
        if training_args.noise_sigma_clipvals != 0.0
        else training_args.nso_sigma_clipvals
    )
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    if model_args.share_embedding and hasattr(model, "tie_weights"):
        model.tie_weights()

    if not model_args.contain_weight_clip_val:
        for name, param in model.named_parameters():
            if "weight_clip_val" in name:
                weight_name = name.replace("weight_clip_val", "weight")
                weight_param = dict(model.named_parameters()).get(weight_name, None)

                if model_args.w_bits == 1:
                    scale = torch.mean(
                        weight_param.abs(), dim=-1, keepdim=True
                    ).detach()
                elif model_args.w_bits == 0 or model_args.w_bits == 2:
                    scale, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                elif model_args.w_bits == 3 or model_args.w_bits == 4:
                    xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    maxq = 2 ** (model_args.w_bits - 1) - 1
                    scale = xmax / maxq
                else:
                    raise NotImplementedError

                param.data.copy_(scale)

    model.cuda()
    log.info("Complete model loading...")

    log.info("Start to load tokenizer...")
    tokenizer = transformers.LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        add_bos_token=model_args.add_bos_token,
        add_eos_token=model_args.add_eos_token,
    )
    log.info("Complete tokenizer loading...")

    train_dataset, valid_dataset = datautils.get_train_val_dataset(
        train_path=data_args.train_data_local_path,
        valid_path=data_args.eval_data_local_path
        if data_args.eval_data_local_path is not None
        else None,
    )
    train_data = datautils.CustomJsonDataset(
        train_dataset, tokenizer, block_size=training_args.model_max_length
    )
    valid_data = datautils.CustomJsonDataset(
        valid_dataset, tokenizer, block_size=min(training_args.model_max_length, 1024)
    )
    model.config.use_cache = False

    callbacks = []
    if (
        training_args.reinitialize_weights
        and training_args.reinitialize_steps
        and training_args.reinitialize_steps > 0
    ):
        callbacks.append(ReinitializeCallback(training_args))

    myTrainer = Trainer
    trainer = myTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_state()
        utils.safe_save_model_for_hf_trainer(
            trainer, model_args.output_model_local_path
        )

    # Evaluation
    if training_args.do_eval:
        model.to("cuda")
        metrics = trainer.evaluate()
        max_eval_samples = len(valid_data)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_data))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    torch.distributed.barrier()


if __name__ == "__main__":
    train()
