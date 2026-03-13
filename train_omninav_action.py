#!/usr/bin/env python3
"""
Training script for OmniNavBench Action prediction with LoRA.
Uses text generation to predict discrete actions.

Usage:
    1. First run preprocess_omninav_actions.py to generate preprocessed data
    2. Then run this script with the preprocessed file
"""

import os
import sys
import math
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import transformers
from transformers import Trainer

from uninavid.constants import IGNORE_INDEX
from uninavid.train.llava_trainer import LLaVATrainer
from uninavid.model import *
from uninavid import conversation as conversation_lib

from uninavid.train.omninav_action_dataset import (
    OmniNavActionDataArguments,
    OmniNavActionDataset,
    ACTION_LABELS,
)


local_rank = None


def rank0_print(*args):
    """Print only on rank 0."""
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    compress_type: Optional[str] = field(default=None)
    run_type: Optional[str] = field(default="train")


@dataclass
class DataArguments:
    """Arguments for data loading."""
    action_root: str = field(default=None)  # Required: OmniNavBenchActionData root
    video_root: str = field(default=None)   # Required: OmniNavBenchVideos root
    split: str = field(default='train')
    inst_types: str = field(default='original,concise,first_person,verbose')  # Comma-separated instruction types
    agent_types: str = field(default='car,dog,human')  # Comma-separated agent types
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_aspect_ratio: str = 'square'
    max_frames: int = field(default=16)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Training arguments."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(default=512)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    lr_multi: Optional[str] = field(default=None)
    tune_vision_encoder: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    """Handle DeepSpeed zero3."""
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    """Get PEFT state dict."""
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """Get non-LoRA state dict."""
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, name=k).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    """Find all linear layer names for LoRA."""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save model safely for HF trainer."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


@dataclass
class DataCollatorForOmniNavAction:
    """Collate examples for OmniNavBench action training."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'prompt' in instances[0]:
            batch['prompts'] = [instance['prompt'] for instance in instances]

        return batch


def make_omninav_action_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    model_args: ModelArguments,
) -> Dict:
    """Make data module for OmniNavBench action training."""
    # Parse instruction types and agent types
    inst_types = [t.strip() for t in data_args.inst_types.split(',')]
    agent_types = [t.strip() for t in data_args.agent_types.split(',')]

    # Create data arguments
    omninav_data_args = OmniNavActionDataArguments(
        action_root=data_args.action_root,
        video_root=data_args.video_root,
        split=data_args.split,
        inst_types=inst_types,
        agent_types=agent_types,
        max_frames=data_args.max_frames,
        image_aspect_ratio=data_args.image_aspect_ratio,
        mm_use_im_start_end=model_args.mm_use_im_start_end,
        is_multimodal=data_args.is_multimodal,
    )

    # Create dataset
    train_dataset = OmniNavActionDataset(
        tokenizer=tokenizer,
        data_args=omninav_data_args,
    )

    # Create data collator
    data_collator = DataCollatorForOmniNavAction(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


def train():
    """Main training function."""
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Validate required arguments
    if not data_args.action_root:
        raise ValueError("--action_root is required (path to OmniNavBenchActionData).")
    if not data_args.video_root:
        raise ValueError("--video_root is required (path to OmniNavBenchVideos).")

    compute_dtype = (
        torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = dict(
        torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)),
    )

    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type
            )
        ))

    rank0_print(f"Loading model from {model_args.model_name_or_path}")

    # Load config
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}
    orig_rope_scaling_factor = orig_rope_scaling.get("factor", 1)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model
    is_llava_model = getattr(config, 'mm_vision_tower', None) is not None or 'vid' in model_args.model_name_or_path.lower()

    if is_llava_model:
        model = LlavaLlamaAttForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = compute_dtype
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )

    # Set conversation template
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["imgsp_v1"]

    # Initialize vision tower if needed
    if is_llava_model and hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=compute_dtype, device=training_args.device)

        # Get image processor
        image_processor = vision_tower.image_processor

        # Update data args with image processor
        data_args.image_processor = image_processor
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

    # Create data module
    data_module = make_omninav_action_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
    )

    # Update data args in dataset
    if hasattr(data_module['train_dataset'], 'data_args'):
        data_module['train_dataset'].data_args.image_processor = data_args.image_processor

    # Create trainer
    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # Save model
    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
