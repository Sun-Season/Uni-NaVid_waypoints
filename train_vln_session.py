#!/usr/bin/env python3
"""
Training script for VLN Session waypoint prediction.
Uses image sequences instead of videos.
"""

import os
import sys
import copy
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import transformers
from transformers import Trainer

from uninavid.constants import IGNORE_INDEX
from uninavid.train.llava_trainer import LLaVATrainer
from uninavid.model import *
from uninavid.model.waypoint_head import LlavaWaypointForCausalLM, WaypointConfig
from uninavid import conversation as conversation_lib

# Import VLN Session dataset
from uninavid.train.vln_session_dataset import (
    VLNSessionDataArguments,
    make_vln_session_data_module
)


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
    run_type: Optional[str] = field(default="train")  # train / eval

    # Waypoint prediction head
    num_future_waypoints: int = field(default=5)


@dataclass
class DataArguments:
    """Arguments for data loading."""
    data_path: str = field(
        default=None,
        metadata={"help": "Path to VLN session directory or parent directory containing multiple sessions"}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    # image_processor is defined in ModelArguments

    # VLN Session specific
    max_frames: int = field(default=32, metadata={"help": "Maximum number of frames to sample"})
    # num_future_waypoints is defined in ModelArguments to avoid conflict
    waypoint_stride: int = field(default=5, metadata={"help": "Stride for sampling waypoints"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Training arguments."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    lr_multi: Optional[str] = field(default=None)

    # Waypoint loss weights
    waypoint_position_weight: float = field(default=1.0)
    waypoint_yaw_weight: float = field(default=1.0)
    waypoint_arrive_weight: float = field(default=1.0)


def rank0_print(*args):
    """Print only on rank 0."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


class WaypointTrainer(LLaVATrainer):
    """Custom trainer for waypoint prediction."""

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss including waypoint prediction losses.
        """
        # Get waypoint targets
        waypoint_positions = inputs.pop("waypoint_positions", None)
        waypoint_yaws = inputs.pop("waypoint_yaws", None)
        waypoint_arrive = inputs.pop("waypoint_arrive", None)
        prompts = inputs.pop("prompts", None)

        # Forward pass
        outputs = model(
            **inputs,
            prompts=prompts,
            return_dict=True
        )

        # Language modeling loss
        lm_loss = outputs.loss if hasattr(outputs, 'loss') else None

        # Waypoint prediction losses
        total_loss = 0.0
        loss_dict = {}

        if lm_loss is not None:
            total_loss += lm_loss
            loss_dict['lm_loss'] = lm_loss.item()

        if hasattr(outputs, 'waypoint_positions') and waypoint_positions is not None:
            # Position loss (MSE)
            pos_loss = torch.nn.functional.mse_loss(
                outputs.waypoint_positions,
                waypoint_positions
            )
            total_loss += self.args.waypoint_position_weight * pos_loss
            loss_dict['waypoint_pos_loss'] = pos_loss.item()

            # Yaw loss (cosine similarity)
            if hasattr(outputs, 'waypoint_yaws') and waypoint_yaws is not None:
                # Normalize predictions and targets
                pred_yaws_norm = torch.nn.functional.normalize(outputs.waypoint_yaws, dim=-1)
                target_yaws_norm = torch.nn.functional.normalize(waypoint_yaws, dim=-1)

                # Cosine similarity loss (1 - cosine_similarity)
                yaw_loss = 1.0 - (pred_yaws_norm * target_yaws_norm).sum(dim=-1).mean()
                total_loss += self.args.waypoint_yaw_weight * yaw_loss
                loss_dict['waypoint_yaw_loss'] = yaw_loss.item()

            # Arrive loss (BCE)
            if hasattr(outputs, 'waypoint_arrive') and waypoint_arrive is not None:
                arrive_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs.waypoint_arrive,
                    waypoint_arrive
                )
                total_loss += self.args.waypoint_arrive_weight * arrive_loss
                loss_dict['waypoint_arrive_loss'] = arrive_loss.item()

        # Log losses
        if self.state.global_step % 10 == 0:
            rank0_print(f"Step {self.state.global_step}: {loss_dict}")

        return (total_loss, outputs) if return_outputs else total_loss


def train():
    """Main training function."""
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    compute_dtype = (
        torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
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

    # Load model
    rank0_print(f"Loading model from {model_args.model_name_or_path}")

    # Load base config and convert to WaypointConfig
    base_config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    
    # Create WaypointConfig from base config
    config = WaypointConfig.from_dict(base_config.to_dict())
    config.num_waypoints = model_args.num_future_waypoints
    config.waypoint_loss_weight = training_args.waypoint_position_weight
    config.angle_loss_weight = training_args.waypoint_yaw_weight
    config.arrive_loss_weight = training_args.waypoint_arrive_weight
    
    # Handle MPT model specific config
    if 'mpt' in model_args.model_name_or_path:
        config.attn_config['attn_impl'] = training_args.mpt_attn_impl

    model = LlavaWaypointForCausalLM.from_pretrained(
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
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
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

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Create VLN Session data module
    vln_data_args = VLNSessionDataArguments(
        data_path=data_args.data_path,
        max_frames=data_args.max_frames,
        num_future_waypoints=model_args.num_future_waypoints,
        waypoint_stride=data_args.waypoint_stride,
        image_processor=data_args.image_processor,
        mm_use_im_start_end=model_args.mm_use_im_start_end,
        is_multimodal=data_args.is_multimodal
    )

    data_module = make_vln_session_data_module(
        tokenizer=tokenizer,
        data_args=vln_data_args
    )

    # Initialize trainer
    trainer = WaypointTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

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
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """Get non-LoRA state dict."""
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v).cpu() for k, v in to_return.items()}
    return to_return


def maybe_zero_3(param):
    """Handle DeepSpeed zero3."""
    if hasattr(param, "ds_id"):
        import deepspeed
        with deepspeed.zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


if __name__ == "__main__":
    train()
