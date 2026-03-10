# Waypoint prediction head for Uni-NaVid
# Adds MLP-based waypoint prediction on top of the VLM

from typing import List, Optional, Tuple, Union, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput

from uninavid.model.language_model.llava_llama_vid import (
    LlavaConfig, LlavaAttLlamaModel, LlavaLlamaAttForCausalLM
)


@dataclass
class WaypointPredictionOutput(ModelOutput):
    """Output class for waypoint prediction model."""
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    waypoint_loss: Optional[torch.FloatTensor] = None
    angle_loss: Optional[torch.FloatTensor] = None
    arrive_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    waypoint_positions: Optional[torch.FloatTensor] = None
    waypoint_yaws: Optional[torch.FloatTensor] = None
    waypoint_arrive: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class WaypointConfig(LlavaConfig):
    """Configuration for waypoint prediction model."""
    model_type = "llava_waypoint"
    
    def __init__(
        self,
        num_waypoints: int = 5,
        waypoint_loss_weight: float = 1.0,
        angle_loss_weight: float = 0.5,
        arrive_loss_weight: float = 0.5,
        use_lm_loss: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_waypoints = num_waypoints
        self.waypoint_loss_weight = waypoint_loss_weight
        self.angle_loss_weight = angle_loss_weight
        self.arrive_loss_weight = arrive_loss_weight
        self.use_lm_loss = use_lm_loss


class WaypointHead(nn.Module):
    """
    MLP-based waypoint prediction head.
    Uses cross-attention to extract action features from VLM hidden states.

    Predicts (r, sin, cos) jointly and computes (x, y) = (r * cos(yaw), r * sin(yaw))
    to ensure position and angle are consistent.
    """

    def __init__(
        self,
        hidden_size: int,
        num_waypoints: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_waypoints = num_waypoints

        # Learnable query for action extraction
        self.query_action = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.normal_(self.query_action, std=0.02)

        # Cross-attention to extract action features from VLM hidden states
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm after attention
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Unified MLP for waypoint prediction: (r, sin, cos) per waypoint
        # Output: [N * 3] -> r, sin, cos for each waypoint
        self.waypoint_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_waypoints * 3)  # [N, 3] for (r, sin, cos)
        )

        # MLP for arrive prediction
        self.arrive_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_waypoints)  # [N] for arrive probability
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for waypoint prediction.

        Predicts (r, sin, cos) jointly and computes (x, y) = (r * cos(yaw), r * sin(yaw)).

        Args:
            hidden_states: [batch_size, seq_len, hidden_size] from VLM
            attention_mask: [batch_size, seq_len] attention mask

        Returns:
            positions: [batch_size, num_waypoints, 2] predicted (x, y)
            angles: [batch_size, num_waypoints, 2] predicted (sin, cos)
            arrive: [batch_size, num_waypoints] predicted arrive probability
        """
        batch_size = hidden_states.shape[0]

        # Expand query for batch
        query = self.query_action.expand(batch_size, -1, -1)

        # Create key padding mask for attention (True = ignore)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        # Cross-attention: query attends to hidden states
        action_features, _ = self.cross_attention(
            query=query,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )

        # Layer norm
        action_features = self.layer_norm(action_features)

        # Squeeze the sequence dimension (we only have 1 query)
        action_features = action_features.squeeze(1)  # [batch_size, hidden_size]

        # Predict (r, sin, cos) jointly
        waypoint_raw = self.waypoint_predictor(action_features)  # [batch_size, N * 3]
        waypoint_raw = waypoint_raw.view(batch_size, self.num_waypoints, 3)  # [batch_size, N, 3]

        # Extract r, sin, cos
        r_raw = waypoint_raw[..., 0]  # [batch_size, N]
        sin_raw = waypoint_raw[..., 1]  # [batch_size, N]
        cos_raw = waypoint_raw[..., 2]  # [batch_size, N]

        # Apply softplus to r to ensure positive
        r = F.softplus(r_raw)  # [batch_size, N]

        # Normalize (sin, cos) to unit circle
        sin_cos = torch.stack([sin_raw, cos_raw], dim=-1)  # [batch_size, N, 2]
        angles_norm = F.normalize(sin_cos, dim=-1)
        sin_yaw = angles_norm[..., 0]  # [batch_size, N]
        cos_yaw = angles_norm[..., 1]  # [batch_size, N]

        # Compute (x, y) from (r, yaw): x = r * cos(yaw), y = r * sin(yaw)
        x = r * cos_yaw  # [batch_size, N]
        y = r * sin_yaw  # [batch_size, N]
        positions = torch.stack([x, y], dim=-1)  # [batch_size, N, 2]

        # Predict arrive probability
        arrive = self.arrive_predictor(action_features)  # [batch_size, num_waypoints]

        return positions, angles_norm, arrive


class LlavaWaypointForCausalLM(LlavaLlamaAttForCausalLM):
    """
    Uni-NaVid model with waypoint prediction head.
    Extends LlavaLlamaAttForCausalLM with MLP-based waypoint prediction.
    """
    config_class = WaypointConfig
    
    def __init__(self, config: WaypointConfig):
        super().__init__(config)
        
        # Waypoint prediction head
        self.waypoint_head = WaypointHead(
            hidden_size=config.hidden_size,
            num_waypoints=config.num_waypoints,
            num_heads=4,
            dropout=0.1
        )
        
        # Loss weights
        self.waypoint_loss_weight = config.waypoint_loss_weight
        self.angle_loss_weight = config.angle_loss_weight
        self.arrive_loss_weight = config.arrive_loss_weight
        self.use_lm_loss = config.use_lm_loss
        
        # Initialize waypoint head
        self._init_waypoint_head()
    
    def _init_waypoint_head(self):
        """Initialize waypoint head weights."""
        for module in self.waypoint_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
        # Waypoint targets
        waypoint_positions: Optional[torch.FloatTensor] = None,
        waypoint_yaws: Optional[torch.FloatTensor] = None,
        waypoint_arrive: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, WaypointPredictionOutput]:
        """
        Forward pass with waypoint prediction.
        
        Additional Args:
            waypoint_positions: [batch_size, num_waypoints, 2] GT positions (x, y)
            waypoint_yaws: [batch_size, num_waypoints, 2] GT angles (sin, cos)
            waypoint_arrive: [batch_size, num_waypoints] GT arrive labels
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = True  # Need hidden states for waypoint prediction
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Move inputs to device if needed
        if not self.training:
            if images is not None and images[0].device != self.device:
                images[0] = images[0].to(device=self.device)
            if input_ids is not None and input_ids.device != self.device:
                input_ids = input_ids.to(device=self.device)
        
        # Prepare multimodal inputs
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images, prompts=prompts
            )

        # Forward through LLM
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]

        # 仅在需要语言建模损失时才计算 lm_head，避免无用的大矩阵乘法
        if self.use_lm_loss:
            logits = self.lm_head(hidden_states)
        else:
            logits = None
        
        # Waypoint prediction
        pred_positions, pred_angles, pred_arrive = self.waypoint_head(
            hidden_states, attention_mask
        )
        
        # Compute losses
        loss = None
        lm_loss = None
        waypoint_loss = None
        angle_loss = None
        arrive_loss = None
        
        # Language modeling loss
        if labels is not None and self.use_lm_loss:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            
            lm_loss = loss_fct(shift_logits, shift_labels)
        
        # Waypoint losses
        if waypoint_positions is not None:
            waypoint_positions = waypoint_positions.to(pred_positions.device)
            waypoint_loss = F.l1_loss(pred_positions, waypoint_positions)
        
        if waypoint_yaws is not None:
            waypoint_yaws = waypoint_yaws.to(pred_angles.device)
            # Cosine similarity loss for angles
            pred_angles_norm = F.normalize(pred_angles, dim=-1)
            gt_angles_norm = F.normalize(waypoint_yaws, dim=-1)
            angle_loss = 1 - (pred_angles_norm * gt_angles_norm).sum(dim=-1).mean()
        
        if waypoint_arrive is not None:
            waypoint_arrive = waypoint_arrive.to(pred_arrive.device)
            arrive_loss = F.binary_cross_entropy_with_logits(pred_arrive, waypoint_arrive)
        
        # Combine losses
        if lm_loss is not None or waypoint_loss is not None:
            loss = 0.0
            if lm_loss is not None:
                loss = loss + lm_loss
            if waypoint_loss is not None:
                loss = loss + self.waypoint_loss_weight * waypoint_loss
            if angle_loss is not None:
                loss = loss + self.angle_loss_weight * angle_loss
            if arrive_loss is not None:
                loss = loss + self.arrive_loss_weight * arrive_loss
        
        if not return_dict:
            output = (logits, pred_positions, pred_angles, pred_arrive) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return WaypointPredictionOutput(
            loss=loss,
            lm_loss=lm_loss,
            waypoint_loss=waypoint_loss,
            angle_loss=angle_loss,
            arrive_loss=arrive_loss,
            logits=logits,
            waypoint_positions=pred_positions,
            waypoint_yaws=pred_angles,
            waypoint_arrive=pred_arrive,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def predict_waypoints(
        self,
        input_ids: torch.LongTensor,
        images: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference method for waypoint prediction.
        
        Returns:
            dict with 'positions', 'angles', 'arrive' tensors
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                prompts=prompts,
                return_dict=True
            )
        
        # Convert arrive logits to probabilities
        arrive_probs = torch.sigmoid(outputs.waypoint_arrive)
        
        # Convert sin/cos to angles
        angles_rad = torch.atan2(
            outputs.waypoint_yaws[..., 0],  # sin
            outputs.waypoint_yaws[..., 1]   # cos
        )
        
        return {
            'positions': outputs.waypoint_positions,  # [batch, N, 2]
            'angles': angles_rad,  # [batch, N] in radians
            'arrive': arrive_probs,  # [batch, N]
            'sin_cos': outputs.waypoint_yaws,  # [batch, N, 2]
        }


# Register the new model
AutoConfig.register("llava_waypoint", WaypointConfig)
AutoModelForCausalLM.register(WaypointConfig, LlavaWaypointForCausalLM)
