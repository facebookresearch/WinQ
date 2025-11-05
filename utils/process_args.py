# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    local_dir: str = field(
        default=None, metadata={"help": "Local Path of storing inputs and outputs "}
    )
    input_model_filename: Optional[str] = field(
        default="test-input", metadata={"help": "Input model relative manifold path"}
    )
    checkpoint_path: Optional[str] = field(
        default=None, metadata={"help": "Optional checkpoint to resume from"}
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    synthesis_path: Optional[str] = field(
        default=None, metadata={"help": "Destination for synthetic dataset outputs"}
    )
    output_model_local_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )
    backend: Optional[str] = field(
        default=None, metadata={"help": "Training backend identifier (for logging)"}
    )
    resume: bool = field(
        default=False, metadata={"help": "Whether to resume from the provided checkpoint"}
    )
    data_path: Optional[str] = field(
        default=None, metadata={"help": "Remote training dataset path"}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Remote evaluation dataset path"}
    )
    use_mdl: bool = field(
        default=False, metadata={"help": "Flag for MDL input pipelines"}
    )
    namespace: Optional[str] = field(
        default=None, metadata={"help": "Namespace for platform integrations"}
    )
    hive_table: Optional[str] = field(
        default=None, metadata={"help": "Hive table name when reading MDL data"}
    )
    hive_json_data_column: Optional[str] = field(
        default=None, metadata={"help": "JSON column name in Hive table"}
    )
    single_file: bool = field(
        default=False, metadata={"help": "Whether to treat dataset inputs as a single file"}
    )
    pretrain: bool = field(
        default=False, metadata={"help": "Legacy pretraining compatibility flag"}
    )
    max_parallel_files: int = field(
        default=1, metadata={"help": "Maximum number of files to read in parallel"}
    )
    w_bits: Optional[int] = field(
        default=32,
        metadata={
            "help": "#bits to use for quantization; use 16 for evaluating base model. choices=[4, 8, 32]"
        },
    )
    a_bits: Optional[int] = field(
        default=16,
        metadata={"help": "#bits for activation quantization (placeholder for compatibility)"},
    )
    kv_bits: Optional[int] = field(
        default=32,
        metadata={"help": "#bits for KV cache quantization (placeholder for compatibility)"},
    )
    contain_weight_clip_val: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set contain_weight_clip_val=True when load a trained quantized model."
        },
    )
    share_embedding: bool = field(
        default=False,
        metadata={"help": "Whether to tie input and output token embeddings"},
    )
    wrap_layer_xl: bool = field(
        default=False,
        metadata={"help": "Compatibility flag for XL layer wrappers"},
    )
    add_bos_token: bool = field(
        default=False,
        metadata={"help": "Override tokenizer BOS token addition"},
    )
    add_eos_token: bool = field(
        default=False,
        metadata={"help": "Override tokenizer EOS token addition"},
    )


@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    train_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Train data local path"}
    )
    eval_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Eval data local path"}
    )
    zeroshot_tasks: Optional[str] = field(
        default=None, metadata={"help": "Comma separated zero-shot eval task list"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    qat: Optional[str] = field(
        default="false",
        metadata={"help": "Enable QAT (True/False/experimental)"},
    )
    train_nso: Optional[bool] = field(default=False)
    nso_within_model: Optional[bool] = field(default=False)
    nso_pre_quantization_noise: Optional[bool] = field(default=False)
    nso_post_quantization_noise: Optional[bool] = field(default=False)
    nso_sigma_weights: Optional[float] = field(default=0.0)
    nso_sigma_clipvals: Optional[float] = field(default=0.0)
    nso_num_perturbs: Optional[int] = field(default=1)
    nso_initialize_noise: Optional[bool] = field(default=False)
    nso_trainable_noise_scale: Optional[bool] = field(default=False)
    noise_injection: Optional[bool] = field(default=False)
    noise_sigma_weights: Optional[float] = field(default=0.0)
    noise_sigma_clipvals: Optional[float] = field(default=0.0)
    pre_quantization_noise: Optional[bool] = field(default=False)
    post_quantization_noise: Optional[bool] = field(default=False)
    initialize_noise: Optional[bool] = field(default=False)
    trainable_noise_scale: Optional[bool] = field(default=False)
    reinitialize_weights: Optional[bool] = field(default=False)
    reinitialize_steps: Optional[int] = field(default=0)
    reinitialize_alpha: Optional[float] = field(default=0.2)
    eval_strategy: Optional[str] = field(
        default=None, metadata={"help": "Alias for evaluation_strategy"}
    )
    warmup_step: Optional[int] = field(
        default=None, metadata={"help": "Alias for warmup_steps"}
    )
    zeroshot_tasks: Optional[str] = field(
        default=None, metadata={"help": "Zero-shot evaluation tasks"}
    )


def _str_to_bool_flag(value: Optional[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"false", "0", "no", "none"}


def process_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(model_args.local_dir, exist_ok=True)

    assert model_args.output_model_local_path is None

    if training_args.eval_strategy is not None:
        training_args.evaluation_strategy = training_args.eval_strategy

    if training_args.warmup_step is not None:
        training_args.warmup_steps = training_args.warmup_step

    if data_args.zeroshot_tasks is None and training_args.zeroshot_tasks is not None:
        data_args.zeroshot_tasks = training_args.zeroshot_tasks

    if data_args.train_data_local_path is None and model_args.data_path is not None:
        data_args.train_data_local_path = model_args.data_path

    if data_args.eval_data_local_path is None and model_args.eval_data_path is not None:
        data_args.eval_data_local_path = model_args.eval_data_path

    training_args.qat = _str_to_bool_flag(training_args.qat)

    model_args.output_model_local_path = os.path.join(
        model_args.local_dir, "models", str(model_args.output_model_filename)
    )

    return model_args, data_args, training_args
