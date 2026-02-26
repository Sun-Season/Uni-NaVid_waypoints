# coding: utf-8
"""
Waypoint-based inference for Uni-NaVid with Waypoint Head.
This module provides inference capabilities for the waypoint prediction model.
"""

import os
import json
import cv2
import numpy as np
import imageio
import torch
import time
import argparse
from typing import List, Dict, Optional, Tuple

from uninavid.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, 
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from uninavid.conversation import conv_templates
from uninavid.mm_utils import tokenizer_image_token


# Set random seeds for reproducibility
seed = 30
torch.manual_seed(seed)
np.random.seed(seed)


def load_waypoint_model(model_path: str, device: str = "cuda"):
    """
    Load the waypoint prediction model.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
    
    Returns:
        tokenizer, model, image_processor, context_len
    """
    from transformers import AutoTokenizer, AutoConfig
    from uninavid.model.waypoint_head import LlavaWaypointForCausalLM, WaypointConfig
    
    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Check if this is a waypoint model or need to convert
    if not hasattr(config, 'num_waypoints'):
        # Convert regular config to waypoint config
        config = WaypointConfig.from_pretrained(model_path)
        config.num_waypoints = 5
        config.waypoint_loss_weight = 1.0
        config.angle_loss_weight = 0.5
        config.arrive_loss_weight = 0.5
        config.use_lm_loss = False  # Don't need LM loss for inference
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Load model
    model = LlavaWaypointForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Load image processor from the model's vision tower to guarantee preprocess parity
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    
    context_len = config.max_position_embeddings
    
    return tokenizer, model, image_processor, context_len


class WaypointAgent:
    """
    Agent for waypoint-based navigation using Uni-NaVid with Waypoint Head.
    Outputs continuous waypoints instead of discrete actions.
    """
    
    def __init__(self, model_path: str, num_waypoints: int = 5):
        """
        Initialize the waypoint agent.
        
        Args:
            model_path: Path to the model checkpoint
            num_waypoints: Number of waypoints to predict
        """
        print("Initialize WaypointAgent")
        
        self.conv_mode = "vicuna_v1"
        self.num_waypoints = num_waypoints
        self.model_path = model_path
        
        # Load model
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            self._load_model(model_path)
        
        assert self.image_processor is not None
        
        print("Initialization Complete")
        
        # Prompt template for waypoint prediction
        self.prompt_template = (
            "Imagine you are a robot programmed for navigation tasks. "
            "You have been given a video of historical observations and an image of the current observation <image>. "
            "Your assigned task is: '{}'. "
            "Predict the next {} waypoints to navigate towards the goal."
        )
        
        self.rgb_list = []
        self.count_id = 0
        self.reset()
    
    def _load_model(self, model_path: str):
        """Load the model with waypoint head support."""
        return load_waypoint_model(model_path)
    
    def process_images(self, rgb_list: List[np.ndarray]) -> List[torch.Tensor]:
        """Process a list of RGB images for model input."""
        batch_image = np.asarray(rgb_list)
        
        if hasattr(self.model, 'get_model'):
            self.model.get_model().new_frames = len(rgb_list)
        
        video = self.image_processor.preprocess(
            batch_image, return_tensors='pt'
        )['pixel_values'].half().cuda()
        
        return [video]
    
    def _prepare_input_ids(self, prompt: str) -> torch.Tensor:
        """Prepare input IDs with special tokens."""
        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IMAGE_SEPARATOR = "<image_sep>"
        
        # Tokenize special tokens
        image_start_special_token = self.tokenizer(
            IMAGE_START_TOKEN, return_tensors="pt"
        ).input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(
            IMAGE_END_TOKEN, return_tensors="pt"
        ).input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(
            VIDEO_START_SPECIAL_TOKEN, return_tensors="pt"
        ).input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(
            VIDEO_END_SPECIAL_TOKEN, return_tensors="pt"
        ).input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(
            NAVIGATION_SPECIAL_TOKEN, return_tensors="pt"
        ).input_ids[0][1:].cuda()
        image_separator = self.tokenizer(
            IMAGE_SEPARATOR, return_tensors="pt"
        ).input_ids[0][1:].cuda()
        
        # Prepare prompt
        qs = prompt
        if hasattr(self.model, 'config') and self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')
        
        # Build conversation
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Tokenize and insert special tokens
        token_prompt = tokenizer_image_token(
            prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).cuda()
        
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_separator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)
        return input_ids
    
    def predict_waypoints(self, prompt: str) -> Dict[str, np.ndarray]:
        """
        Predict waypoints given a navigation prompt.
        
        Args:
            prompt: Navigation instruction prompt
        
        Returns:
            Dictionary with:
                - 'positions': [N, 2] array of (x, y) positions
                - 'angles': [N] array of yaw angles in radians
                - 'arrive': [N] array of arrive probabilities
        """
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        
        # Prepare inputs
        input_ids = self._prepare_input_ids(prompt)
        imgs = self.process_images(self.rgb_list)
        self.rgb_list = []
        
        # Update prompt for model
        if hasattr(self.model, 'update_prompt'):
            self.model.update_prompt([[question]])
        
        # Predict waypoints
        with torch.inference_mode():
            if not hasattr(self.model, 'predict_waypoints'):
                raise RuntimeError(
                    "Loaded model does not implement `predict_waypoints`. "
                    "Please load a waypoint-capable checkpoint."
                )

            outputs = self.model.predict_waypoints(
                input_ids=input_ids,
                images=imgs,
                prompts=[question]
            )

            return {
                'positions': outputs['positions'][0].cpu().numpy(),
                'angles': outputs['angles'][0].cpu().numpy(),
                'arrive': outputs['arrive'][0].cpu().numpy(),
                'sin_cos': outputs['sin_cos'][0].cpu().numpy()
            }
    
    def reset(self, task_type: str = 'vln'):
        """Reset the agent state."""
        self.transformation_list = []
        self.rgb_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []
        self.task_type = task_type
        self.first_forward = False
        self.executed_steps = 0
        
        if hasattr(self.model, 'config'):
            self.model.config.run_type = "eval"
        
        if hasattr(self.model, 'get_model'):
            model = self.model.get_model()
            if hasattr(model, 'initialize_online_inference_nav_feat_cache'):
                model.initialize_online_inference_nav_feat_cache()
            if hasattr(model, 'new_frames'):
                model.new_frames = 0
    
    def act(self, data: Dict) -> Dict:
        """
        Perform navigation action given observation data.
        
        Args:
            data: Dictionary with 'observations' (RGB image) and 'instruction'
        
        Returns:
            Dictionary with:
                - 'step': Current step number
                - 'waypoints': Predicted waypoints
                - 'positions': [N, 2] positions
                - 'angles': [N] angles
                - 'arrive': [N] arrive probabilities
        """
        rgb = data["observations"]
        self.rgb_list.append(rgb)
        
        # Build prompt
        navigation_prompt = self.prompt_template.format(
            data["instruction"], 
            self.num_waypoints
        )
        
        # Predict waypoints
        waypoints = self.predict_waypoints(navigation_prompt)
        
        self.executed_steps += 1
        
        # Build trajectory from waypoints
        traj = [[0.0, 0.0, 0.0]]
        for i in range(len(waypoints['positions'])):
            pos = waypoints['positions'][i]
            angle = waypoints['angles'][i]
            traj.append([pos[0], pos[1], angle])
        
        self.latest_action = {
            "step": self.executed_steps,
            "path": [traj],
            "waypoints": waypoints,
            "positions": waypoints['positions'],
            "angles": waypoints['angles'],
            "arrive": waypoints['arrive']
        }
        
        return self.latest_action.copy()


def draw_waypoints_fpv(
    img: np.ndarray,
    positions: np.ndarray,
    angles: np.ndarray,
    arrive: np.ndarray,
    scale: float = 50.0,
    arrow_color: Tuple[int, int, int] = (0, 255, 0),
    arrive_color: Tuple[int, int, int] = (0, 0, 255),
    arrow_thickness: int = 2,
    point_radius: int = 5
) -> np.ndarray:
    """
    Draw waypoints on first-person view image.
    
    Args:
        img: Input image
        positions: [N, 2] waypoint positions (x, y) in meters
        angles: [N] waypoint angles in radians
        arrive: [N] arrive probabilities
        scale: Pixels per meter for visualization
        arrow_color: Color for waypoint arrows
        arrive_color: Color for arrive indicator
        arrow_thickness: Thickness of arrows
        point_radius: Radius of waypoint points
    
    Returns:
        Image with waypoints drawn
    """
    out = img.copy()
    h, w = out.shape[:2]
    
    # Base position (bottom center of image)
    base_x, base_y = w // 2, int(h * 0.9)
    
    prev_point = (base_x, base_y)
    
    for i in range(len(positions)):
        x, y = positions[i]
        angle = angles[i]
        arrive_prob = arrive[i]
        
        # Convert to pixel coordinates
        # x is forward (up in image), y is left/right
        px = int(base_x - y * scale)  # y -> horizontal
        py = int(base_y - x * scale)  # x -> vertical (forward = up)
        
        current_point = (px, py)
        
        # Draw line from previous point
        cv2.line(out, prev_point, current_point, arrow_color, arrow_thickness)
        
        # Draw waypoint point
        if arrive_prob > 0.5:
            # High arrive probability - draw as stop indicator
            cv2.circle(out, current_point, point_radius + 2, arrive_color, -1)
        else:
            cv2.circle(out, current_point, point_radius, arrow_color, -1)
        
        # Draw direction arrow
        arrow_len = 15
        end_x = int(px - arrow_len * np.sin(angle))
        end_y = int(py - arrow_len * np.cos(angle))
        cv2.arrowedLine(
            out, current_point, (end_x, end_y), 
            arrow_color, arrow_thickness, tipLength=0.3
        )
        
        prev_point = current_point
    
    # Convert BGR to RGB for display
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


def get_sorted_images(recording_dir: str) -> List[np.ndarray]:
    """Load and sort images from a recording directory."""
    image_dir = os.path.join(recording_dir, 'images')
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    images = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        np_image = cv2.imread(image_path)
        images.append(np_image)
    
    return images


def get_instruction(recording_dir: str) -> str:
    """Load instruction from recording directory."""
    json_path = os.path.join(recording_dir, "instruction.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        instruction = json.load(f)["instruction"]
    return instruction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waypoint-based navigation inference')
    parser.add_argument('test_case', help='Test case path (images dir)')
    parser.add_argument('output_dir', help='Output directory to save results')
    parser.add_argument('--model_path', default='model_zoo/uninavid-waypoint-7b',
                        help='Path to model checkpoint')
    parser.add_argument('--num_waypoints', type=int, default=5,
                        help='Number of waypoints to predict')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize agent
    agent = WaypointAgent(args.model_path, num_waypoints=args.num_waypoints)
    agent.reset()
    
    # Load test data
    images = get_sorted_images(args.test_case)
    instruction = get_instruction(args.test_case)
    print(f"Total {len(images)} images")
    print(f"Instruction: {instruction}")
    
    # Run inference
    result_vis_list = []
    all_results = []
    
    for i, img in enumerate(images):
        t_start = time.time()
        result = agent.act({'instruction': instruction, 'observations': img})
        inference_time = time.time() - t_start
        
        print(f"Step {result['step']}, inference time: {inference_time:.3f}s")
        print(f"  Positions: {result['positions']}")
        print(f"  Angles: {result['angles']}")
        print(f"  Arrive: {result['arrive']}")
        
        # Visualize
        vis = draw_waypoints_fpv(
            img,
            result['positions'],
            result['angles'],
            result['arrive']
        )
        result_vis_list.append(vis)
        
        # Store results
        all_results.append({
            'step': result['step'],
            'positions': result['positions'].tolist(),
            'angles': result['angles'].tolist(),
            'arrive': result['arrive'].tolist()
        })
    
    # Save results
    imageio.mimsave(os.path.join(args.output_dir, "result.gif"), result_vis_list)
    
    with open(os.path.join(args.output_dir, "waypoints.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {args.output_dir}")
