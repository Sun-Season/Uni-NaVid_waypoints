#!/usr/bin/env python3
"""
Action prediction head for discrete action sequence classification.
Predicts multiple future actions at once.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

from transformers import LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from uninavid.model.language_model.llava_llama_vid import (
    LlavaLlamaAttForCausalLM,
)


# Action labels
ACTION_LABELS = ['STOP', 'FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
NUM_ACTIONS = len(ACTION_LABELS)


class ActionConfig(LlamaConfig):
    """Configuration for action prediction model."""
    model_type = "llava_action"

    def __init__(
        self,
        num_actions: int = NUM_ACTIONS,
        num_future_actions: int = 5,  # Number of future actions to predict
        action_loss_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_actions = num_actions
        self.num_future_actions = num_future_actions
        self.action_loss_weight = action_loss_weight


@dataclass
class ActionModelOutput(CausalLMOutputWithPast):
    """Output of action prediction model."""
    action_logits: Optional[torch.FloatTensor] = None  # [batch, num_future_actions, num_actions]
    action_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None  # 添加lm_loss


class ActionHead(nn.Module):
    """
    Action prediction head using cross-attention.
    Predicts multiple future discrete actions from VLM hidden states.
    """

    def __init__(self, config: ActionConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_actions = config.num_actions
        num_future_actions = config.num_future_actions

        self.num_future_actions = num_future_actions
        self.num_actions = num_actions

        # Learnable queries for each future action
        self.query_actions = nn.Parameter(
            torch.randn(1, num_future_actions, hidden_size) * 0.02
        )

        # Cross-attention to extract action features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Action classifier for each future step
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_actions)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len]

        Returns:
            action_logits: [batch, num_future_actions, num_actions]
        """
        batch_size = hidden_states.shape[0]

        # Expand queries for batch: [batch, num_future_actions, hidden_size]
        queries = self.query_actions.expand(batch_size, -1, -1)

        # Create key padding mask for attention
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        # Cross-attention: queries attend to hidden states
        action_features, _ = self.cross_attention(
            query=queries,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )

        # Layer norm: [batch, num_future_actions, hidden_size]
        action_features = self.layer_norm(action_features)

        # Classify each future action: [batch, num_future_actions, num_actions]
        action_logits = self.classifier(action_features)

        return action_logits


class LlavaActionForCausalLM(LlavaLlamaAttForCausalLM):
    """
    LLaVA model with action prediction head.
    Predicts multiple future actions.
    """
    config_class = ActionConfig

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        self.action_head = ActionHead(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        action_labels: Optional[torch.LongTensor] = None,  # [batch, num_future_actions]
        prompts: Optional[list] = None,
    ) -> ActionModelOutput:
        """Forward pass with action prediction."""

        # Prepare inputs for multimodal
        if inputs_embeds is None and images is not None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                prompts=prompts
            )

        # Forward through LLM
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Compute LM loss
        lm_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)

        # Action prediction: [batch, num_future_actions, num_actions]
        action_logits = self.action_head(hidden_states, attention_mask)

        # Compute action loss (average over all future actions)
        action_loss = None
        if action_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Reshape for loss: [batch * num_future_actions, num_actions]
            batch_size, num_future = action_labels.shape
            action_logits_flat = action_logits.view(-1, self.config.num_actions)
            action_labels_flat = action_labels.view(-1)
            action_loss = loss_fct(action_logits_flat, action_labels_flat)

        # Total loss
        loss = None
        if lm_loss is not None:
            loss = lm_loss
            if action_loss is not None:
                loss = loss + self.config.action_loss_weight * action_loss

        return ActionModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            action_logits=action_logits,
            action_loss=action_loss,
            lm_loss=lm_loss,  # 添加lm_loss到输出
        )

    def predict_actions(
        self,
        input_ids: torch.LongTensor,
        images: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompts: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict multiple future actions for inference.

        Returns:
            action_ids: [num_future_actions] Predicted action indices
            action_probs: [num_future_actions, num_actions] Action probabilities
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                images=images,
                attention_mask=attention_mask,
                prompts=prompts,
                return_dict=True,
            )

            # action_logits: [batch, num_future_actions, num_actions]
            action_logits = outputs.action_logits[0]  # Take first batch
            action_probs = torch.softmax(action_logits, dim=-1)
            action_ids = action_logits.argmax(dim=-1)

        return action_ids, action_probs
