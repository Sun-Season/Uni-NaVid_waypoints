#!/usr/bin/env python3
"""
Offline evaluation script for VLN Action Text model.
Tests the trained model on passed_samples without simulator.
"""

import os
import sys
import json
import math
import argparse
from typing import List, Dict, Tuple
from collections import Counter

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uninavid.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle


# Waypoint to action conversion (same as training)
FORWARD_DISTANCE = 0.25
TURN_ANGLE = math.radians(30)
MIN_DISPLACEMENT = 0.05
NUM_ACTIONS = 4


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def waypoint_to_actions(current_pose: dict, future_pose: dict) -> List[str]:
    """Convert waypoint to discrete actions (ground truth)."""
    dx = future_pose['x'] - current_pose['x']
    dy = future_pose['y'] - current_pose['y']
    r = math.sqrt(dx*dx + dy*dy)
    yaw_diff = normalize_angle(future_pose['yaw'] - current_pose['yaw'])

    actions = []

    if r >= MIN_DISPLACEMENT:
        global_theta = math.atan2(dy, dx)
        relative_theta = normalize_angle(global_theta - current_pose['yaw'])

        while abs(relative_theta) > TURN_ANGLE / 2:
            if relative_theta > 0:
                actions.append('left')
                relative_theta -= TURN_ANGLE
            else:
                actions.append('right')
                relative_theta += TURN_ANGLE

        remaining_dist = r
        while remaining_dist > FORWARD_DISTANCE / 2:
            actions.append('forward')
            remaining_dist -= FORWARD_DISTANCE
    else:
        while abs(yaw_diff) > TURN_ANGLE / 2:
            if yaw_diff > 0:
                actions.append('left')
                yaw_diff -= TURN_ANGLE
            else:
                actions.append('right')
                yaw_diff += TURN_ANGLE

    if len(actions) == 0:
        actions = ['wait'] * NUM_ACTIONS
    else:
        if len(actions) > NUM_ACTIONS:
            actions = actions[:NUM_ACTIONS]
        while len(actions) < NUM_ACTIONS:
            actions.append(actions[-1])

    return actions


class ActionTextEvaluator:
    """Evaluator for VLN Action Text model."""

    def __init__(self, model_path: str, lora_path: str = None):
        print(f"Loading model from {model_path}")
        if lora_path:
            print(f"Loading LoRA from {lora_path}")

        # load_pretrained_model(model_path, model_base, model_name)
        # When lora_path is provided: model_path=lora_path, model_base=base_model
        # model_name must contain 'vid' to load LlavaLlamaAttForCausalLM
        model_name = get_model_name_from_path(model_path)  # Use base model name (contains 'vid')
        if lora_path:
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                lora_path, model_path, model_name
            )
        else:
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path, None, model_name
            )

        self.model.eval()
        self.model.config.run_type = "eval"
        self.conv_mode = "vicuna_v1"

        # Special tokens
        self.VIDEO_START = "<video_special>"
        self.VIDEO_END = "</video_special>"
        self.IMAGE_START = "<image_special>"
        self.IMAGE_END = "</image_special>"
        self.NAV_TOKEN = "[Navigation]"
        self.IMAGE_SEP = "<image_sep>"

        print("Model loaded successfully")

    def reset(self):
        """Reset model state for new episode."""
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

    def process_images(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """Process images for model input."""
        batch_image = np.asarray(images)
        self.model.get_model().new_frames = len(images)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values']
        video = video.half().cuda()
        return [video]

    def predict(self, images: List[np.ndarray], instruction: str) -> str:
        """Predict actions given images and instruction."""
        # Build prompt (matching training format)
        prompt_template = (
            "This is a navigation video. The instruction is: {}\n"
            "Based on the visual observation and instruction, determine your next four actions. "
            "The predicted action should be one of the following: forward, left, right, or wait."
        )
        question = prompt_template.format(instruction)
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        # Build conversation
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize with special tokens
        video_start = self.tokenizer(self.VIDEO_START, return_tensors="pt").input_ids[0][1:].cuda()
        video_end = self.tokenizer(self.VIDEO_END, return_tensors="pt").input_ids[0][1:].cuda()
        image_start = self.tokenizer(self.IMAGE_START, return_tensors="pt").input_ids[0][1:].cuda()
        image_end = self.tokenizer(self.IMAGE_END, return_tensors="pt").input_ids[0][1:].cuda()
        nav_token = self.tokenizer(self.NAV_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_sep = self.tokenizer(self.IMAGE_SEP, return_tensors="pt").input_ids[0][1:].cuda()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]

        new_list = []
        while indices.numel() > 0:
            idx = indices[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start)
            new_list.append(image_sep)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end)
            new_list.append(image_start)
            new_list.append(image_end)
            new_list.append(nav_token)
            token_prompt = token_prompt[idx + 1:]
            indices = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]

        if token_prompt.numel() > 0:
            new_list.append(token_prompt)

        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        # Stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        # Process images
        imgs = self.process_images(images)

        # Generate
        with torch.inference_mode():
            self.model.update_prompt([[question]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=False,  # Greedy for evaluation
                temperature=0.0,
                max_new_tokens=32,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        # Decode
        input_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]

        return outputs.strip()

    def parse_actions(self, output: str) -> List[str]:
        """Parse model output to action list."""
        valid_actions = {'forward', 'left', 'right', 'wait', 'stop'}
        actions = []
        for word in output.lower().split():
            if word in valid_actions:
                # Map 'stop' to 'wait' for consistency
                actions.append('wait' if word == 'stop' else word)

        # Pad or truncate to NUM_ACTIONS
        if len(actions) == 0:
            actions = ['wait'] * NUM_ACTIONS
        elif len(actions) < NUM_ACTIONS:
            while len(actions) < NUM_ACTIONS:
                actions.append(actions[-1])
        else:
            actions = actions[:NUM_ACTIONS]

        return actions


def load_session_data(session_path: str) -> Tuple[List[dict], str]:
    """Load trajectory and instruction from session."""
    with open(os.path.join(session_path, 'trajectory.json')) as f:
        traj_data = json.load(f)

    trajectory = traj_data['trajectory']

    instruction = ""
    inst_file = os.path.join(session_path, 'instructions.json')
    if os.path.exists(inst_file):
        with open(inst_file) as f:
            inst_data = json.load(f)
        if 'instructions' in inst_data and 'description_overall' in inst_data['instructions']:
            instruction = inst_data['instructions']['description_overall']

    return trajectory, instruction


def evaluate_sample(
    evaluator: ActionTextEvaluator,
    session_path: str,
    trajectory: List[dict],
    instruction: str,
    frame_idx: int,
    stride: int,
    max_frames: int = 16
) -> Dict:
    """Evaluate a single sample."""
    num_frames = len(trajectory)

    # Load history frames
    start_idx = max(0, frame_idx - max_frames + 1)
    frame_indices = list(range(start_idx, frame_idx + 1))

    images = []
    for idx in frame_indices:
        img_file = trajectory[idx]['image_file']
        img_path = os.path.join(session_path, img_file)
        img = Image.open(img_path).convert('RGB')
        images.append(np.array(img))

    # Get ground truth
    current_pose = trajectory[frame_idx]['pose']
    future_idx = min(frame_idx + stride, num_frames - 1)
    future_pose = trajectory[future_idx]['pose']
    gt_actions = waypoint_to_actions(current_pose, future_pose)

    # Predict
    evaluator.reset()
    output = evaluator.predict(images, instruction)
    pred_actions = evaluator.parse_actions(output)

    # Calculate metrics
    correct = sum(1 for p, g in zip(pred_actions, gt_actions) if p == g)
    exact_match = pred_actions == gt_actions
    first_correct = pred_actions[0] == gt_actions[0] if pred_actions and gt_actions else False

    return {
        'session': os.path.basename(session_path),
        'frame_idx': frame_idx,
        'gt_actions': gt_actions,
        'pred_actions': pred_actions,
        'raw_output': output,
        'correct_count': correct,
        'exact_match': exact_match,
        'first_correct': first_correct,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate VLN Action Text model')
    parser.add_argument('--model_path', type=str,
                        default='/mnt/dataset/wj_zqc/VLN/model/uninavid-7b-full-224-video-fps-1-grid-2',
                        help='Base model path')
    parser.add_argument('--lora_path', type=str,
                        default='output/vln_action_text_test',
                        help='LoRA checkpoint path')
    parser.add_argument('--data_path', type=str,
                        default='passed_samples',
                        help='Path to test data')
    parser.add_argument('--stride', type=int, default=60,
                        help='Frame stride for waypoint sampling')
    parser.add_argument('--max_frames', type=int, default=16,
                        help='Maximum history frames')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to evaluate (None for all)')
    parser.add_argument('--output', type=str, default='eval_results.json',
                        help='Output file for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results')
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ActionTextEvaluator(args.model_path, args.lora_path)

    # Find sessions
    session_dirs = []
    for item in sorted(os.listdir(args.data_path)):
        item_path = os.path.join(args.data_path, item)
        if os.path.isdir(item_path):
            traj_file = os.path.join(item_path, 'trajectory.json')
            if os.path.exists(traj_file):
                session_dirs.append(item_path)

    print(f"Found {len(session_dirs)} sessions")

    # Build sample list
    samples = []
    for session_path in session_dirs:
        trajectory, instruction = load_session_data(session_path)
        num_frames = len(trajectory)

        for frame_idx in range(4, num_frames, args.stride):
            samples.append({
                'session_path': session_path,
                'trajectory': trajectory,
                'instruction': instruction,
                'frame_idx': frame_idx,
            })

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Evaluating {len(samples)} samples")
    print()

    # Evaluate
    results = []
    correct_total = 0
    exact_matches = 0
    first_correct_total = 0

    action_confusion = {a: Counter() for a in ['forward', 'left', 'right', 'wait']}

    for sample in tqdm(samples, desc="Evaluating"):
        result = evaluate_sample(
            evaluator,
            sample['session_path'],
            sample['trajectory'],
            sample['instruction'],
            sample['frame_idx'],
            args.stride,
            args.max_frames
        )
        results.append(result)

        correct_total += result['correct_count']
        exact_matches += int(result['exact_match'])
        first_correct_total += int(result['first_correct'])

        # Update confusion matrix
        for gt, pred in zip(result['gt_actions'], result['pred_actions']):
            action_confusion[gt][pred] += 1

        if args.verbose:
            print(f"\n{result['session']} frame {result['frame_idx']}:")
            print(f"  GT:   {' '.join(result['gt_actions'])}")
            print(f"  Pred: {' '.join(result['pred_actions'])}")
            print(f"  Raw:  {result['raw_output']}")
            print(f"  Correct: {result['correct_count']}/4, Exact: {result['exact_match']}")

    # Calculate metrics
    total_actions = len(samples) * NUM_ACTIONS
    action_accuracy = correct_total / total_actions if total_actions > 0 else 0
    exact_match_rate = exact_matches / len(samples) if samples else 0
    first_action_accuracy = first_correct_total / len(samples) if samples else 0

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total samples: {len(samples)}")
    print(f"Action accuracy: {action_accuracy:.2%} ({correct_total}/{total_actions})")
    print(f"Exact match rate: {exact_match_rate:.2%} ({exact_matches}/{len(samples)})")
    print(f"First action accuracy: {first_action_accuracy:.2%} ({first_correct_total}/{len(samples)})")

    print("\nPer-action accuracy:")
    for action in ['forward', 'left', 'right', 'wait']:
        total = sum(action_confusion[action].values())
        correct = action_confusion[action][action]
        acc = correct / total if total > 0 else 0
        print(f"  {action}: {acc:.2%} ({correct}/{total})")

    print("\nConfusion matrix (rows=GT, cols=Pred):")
    actions = ['forward', 'left', 'right', 'wait']
    print("         " + " ".join(f"{a:>8}" for a in actions))
    for gt in actions:
        row = [action_confusion[gt][pred] for pred in actions]
        print(f"{gt:>8} " + " ".join(f"{c:>8}" for c in row))

    # Save results
    output_data = {
        'config': {
            'model_path': args.model_path,
            'lora_path': args.lora_path,
            'stride': args.stride,
            'max_frames': args.max_frames,
        },
        'metrics': {
            'total_samples': len(samples),
            'action_accuracy': action_accuracy,
            'exact_match_rate': exact_match_rate,
            'first_action_accuracy': first_action_accuracy,
        },
        'confusion_matrix': {gt: dict(action_confusion[gt]) for gt in actions},
        'results': results,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
