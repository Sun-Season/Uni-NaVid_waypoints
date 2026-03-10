# coding: utf-8
"""
Waypoint prediction inference script for Uni-NaVid.
Based on offline_eval_uninavid.py but adapted for waypoint prediction.
"""
import os
import json
import cv2
import numpy as np
import imageio
import torch
import time
import argparse

from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    VIDEO_START_SPECIAL_TOKEN, VIDEO_END_SPECIAL_TOKEN,
    IMAGE_START_TOKEN, IMAGE_END_TOKEN, NAVIGATION_SPECIAL_TOKEN,
    IAMGE_SEPARATOR, NAVIGATION_IDENTIFIER
)
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria


seed = 30
torch.manual_seed(seed)
np.random.seed(seed)


class UniNaVid_Waypoint_Agent():
    def __init__(self, model_path, model_base=None):
        """
        Initialize Waypoint prediction agent.

        Args:
            model_path: Path to LoRA checkpoint (e.g., "model_zoo/uninavid-7b-omninav-waypoint/checkpoint-1500")
            model_base: Path to base model (e.g., "lmsys/vicuna-7b-v1.5"). If None, load full model.
        """
        print("Initialize UniNaVid Waypoint Model")
        print(f"Model path: {model_path}")
        if model_base:
            print(f"Base model: {model_base}")

        self.conv_mode = "vicuna_v1"

        # Load model (builder.py handles LoRA loading automatically)
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name
        )

        assert self.image_processor is not None
        print("Initialization Complete")

        self.rgb_list = []
        self.count_id = 0
        self.reset()

    def process_images(self, rgb_list):
        """Process image list into model input format."""
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()
        return [video]

    def predict_waypoints(self, instruction):
        """
        Predict waypoints given instruction and image history.

        Args:
            instruction: Navigation instruction string

        Returns:
            dict with 'positions', 'angles', 'arrive' predictions
        """
        # Build prompt with NAVIGATION_IDENTIFIER
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{NAVIGATION_IDENTIFIER}{instruction}"

        # Tokenize
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # Build input_ids with special tokens
        token_prompt = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

        # Add special tokens for video/image/navigation
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        # Replace IMAGE_TOKEN_INDEX (-200) with special tokens
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
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

        # Process images
        imgs = self.process_images(self.rgb_list)
        self.rgb_list = []

        # Predict waypoints using the model's predict_waypoints method
        with torch.inference_mode():
            self.model.update_prompt([[f"{NAVIGATION_IDENTIFIER}{instruction}"]])
            waypoint_predictions = self.model.predict_waypoints(
                input_ids=input_ids,
                images=imgs,
                prompts=[[f"{NAVIGATION_IDENTIFIER}{instruction}"]]
            )

        return waypoint_predictions

    def reset(self, task_type='vln'):
        """Reset agent state."""
        self.transformation_list = []
        self.rgb_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []
        self.task_type = task_type

        self.first_forward = False
        self.executed_steps = 0
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

    def act(self, data):
        """
        Execute one prediction step.

        Args:
            data: dict with 'observations' (RGB image) and 'instruction' (task instruction)

        Returns:
            dict with 'step', 'waypoints', 'raw_predictions'
        """
        rgb = data["observations"]
        self.rgb_list.append(rgb)

        # Predict waypoints
        waypoint_predictions = self.predict_waypoints(data["instruction"])

        # Extract predictions (batch_size=1)
        positions = waypoint_predictions['positions'][0].cpu().numpy()  # [N, 2]
        angles = waypoint_predictions['angles'][0].cpu().numpy()  # [N]
        arrive = waypoint_predictions['arrive'][0].cpu().numpy()  # [N]

        # Build waypoint list: [(x, y, yaw), ...]
        waypoints = []
        for i in range(len(positions)):
            x, y = positions[i]
            yaw = angles[i]
            waypoints.append([float(x), float(y), float(yaw)])

        self.executed_steps += 1

        self.latest_action = {
            "step": self.executed_steps,
            "waypoints": waypoints,
            "arrive_probs": arrive.tolist(),
            "raw_predictions": {
                "positions": positions.tolist(),
                "angles": angles.tolist(),
                "arrive": arrive.tolist()
            }
        }

        return self.latest_action.copy()


def get_sorted_images(recording_dir):
    """Load sorted images from directory."""
    image_dir = os.path.join(recording_dir, 'images')

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    images = []
    for step, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        np_image = cv2.imread(image_path)
        images.append(np_image)

    return images


def get_traj_data(recording_dir):
    """Load instruction from directory."""
    json_path = os.path.join(recording_dir, "instruction.json")

    with open(json_path, 'r', encoding='utf-8') as f:
        instruction = json.load(f)["instruction"]

    return instruction


def draw_waypoints_on_image(img, waypoints, arrive_probs=None, scale=50, origin_ratio=(0.5, 0.9)):
    """
    Draw predicted waypoints on image.

    Args:
        img: Input image
        waypoints: List of [x, y, yaw] waypoints
        arrive_probs: List of arrive probabilities (optional)
        scale: Scale factor to convert meters to pixels
        origin_ratio: Robot position in image (width_ratio, height_ratio)
    """
    out = img.copy()
    h, w = out.shape[:2]

    # Robot starting position (bottom center)
    origin_x = int(w * origin_ratio[0])
    origin_y = int(h * origin_ratio[1])

    # Draw robot position
    cv2.circle(out, (origin_x, origin_y), 8, (255, 0, 0), -1)  # Blue dot
    cv2.putText(out, "Robot", (origin_x + 10, origin_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Color gradient: green -> yellow -> orange -> red
    colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (255, 0, 255), (0, 0, 255)]

    prev_point = (origin_x, origin_y)

    for i, waypoint in enumerate(waypoints):
        x, y, yaw = waypoint

        # Convert coordinates: x forward (image up), y left (image left)
        img_x = int(origin_x - y * scale)
        img_y = int(origin_y - x * scale)

        # Clamp to image bounds
        img_x = max(0, min(w-1, img_x))
        img_y = max(0, min(h-1, img_y))

        curr_point = (img_x, img_y)

        # Choose color
        color_idx = min(i, len(colors) - 1)
        color = colors[color_idx]

        # Draw line from previous waypoint
        cv2.line(out, prev_point, curr_point, color, 2)

        # Draw waypoint circle
        cv2.circle(out, curr_point, 6, color, -1)

        # Draw orientation arrow
        arrow_len = 20
        arrow_end_x = int(curr_point[0] + arrow_len * np.sin(yaw))
        arrow_end_y = int(curr_point[1] - arrow_len * np.cos(yaw))
        cv2.arrowedLine(out, curr_point, (arrow_end_x, arrow_end_y), color, 2, tipLength=0.4)

        # Label waypoint number
        label = f"{i+1}"
        if arrive_probs is not None:
            label += f" ({arrive_probs[i]:.2f})"
        cv2.putText(out, label, (curr_point[0]+10, curr_point[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        prev_point = curr_point

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waypoint prediction inference')
    parser.add_argument('test_case', help='Test case path (directory with images/)')
    parser.add_argument('output_dir', help='Output directory to save results')
    parser.add_argument('--model_path', default='model_zoo/uninavid-7b-omninav-waypoint/checkpoint-1500',
                       help='Path to LoRA checkpoint')
    parser.add_argument('--model_base', default='lmsys/vicuna-7b-v1.5',
                       help='Path to base model (use "None" to load full model)')

    args = parser.parse_args()

    # Handle "None" string
    if args.model_base == "None":
        args.model_base = None

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize agent
    print(f"\n{'='*60}")
    print("Loading Waypoint Prediction Model")
    print(f"{'='*60}")
    agent = UniNaVid_Waypoint_Agent(args.model_path, args.model_base)
    agent.reset()

    # Load test data
    images = get_sorted_images(args.test_case)
    instruction = get_traj_data(args.test_case)
    print(f"\n{'='*60}")
    print(f"Test Case: {args.test_case}")
    print(f"Instruction: {instruction}")
    print(f"Total images: {len(images)}")
    print(f"{'='*60}\n")

    # Run inference
    result_vis_list = []
    all_results = []
    step_count = 0

    for i, img in enumerate(images):
        print(f"--- Step {i+1}/{len(images)} ---")

        t_s = time.time()
        result = agent.act({'instruction': instruction, 'observations': img})
        inference_time = time.time() - t_s

        step_count += 1
        print(f"Inference time: {inference_time:.3f}s")
        print(f"Predicted waypoints:")
        for j, wp in enumerate(result['waypoints']):
            print(f"  WP{j+1}: x={wp[0]:.3f}m, y={wp[1]:.3f}m, yaw={wp[2]:.3f}rad, arrive={result['arrive_probs'][j]:.3f}")

        # Visualize
        vis = draw_waypoints_on_image(img, result['waypoints'], result['arrive_probs'])
        result_vis_list.append(vis)

        # Save result
        all_results.append({
            'step': step_count,
            'waypoints': result['waypoints'],
            'arrive_probs': result['arrive_probs'],
            'inference_time': inference_time
        })

    # Save visualization GIF
    gif_path = os.path.join(args.output_dir, "waypoints_result.gif")
    imageio.mimsave(gif_path, result_vis_list, duration=0.5)
    print(f"\n{'='*60}")
    print(f"Saved visualization: {gif_path}")

    # Save JSON results
    json_path = os.path.join(args.output_dir, "waypoints_result.json")
    with open(json_path, 'w') as f:
        json.dump({
            'test_case': args.test_case,
            'instruction': instruction,
            'model_path': args.model_path,
            'model_base': args.model_base,
            'results': all_results
        }, f, indent=2)
    print(f"Saved results: {json_path}")
    print(f"{'='*60}\n")

    print("Inference complete!")
