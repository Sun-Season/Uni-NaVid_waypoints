#!/usr/bin/env python3
"""
Online evaluation script for VLN Action Text model.
Simulates continuous navigation with incremental frame input.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uninavid.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle


# Waypoint to action conversion
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
    """Convert waypoint to discrete actions."""
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


class OnlineActionEvaluator:
    """Online evaluator that simulates continuous navigation."""

    def __init__(self, model_path: str, lora_path: str = None):
        print(f"Loading model from {model_path}")
        if lora_path:
            print(f"Loading LoRA from {lora_path}")

        model_name = get_model_name_from_path(model_path)
        if lora_path:
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                lora_path, model_path, model_name
            )
        else:
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path, None, model_name
            )

        self.model.eval()
        self.conv_mode = "vicuna_v1"

        # Special tokens
        self.VIDEO_START = "<video_special>"
        self.VIDEO_END = "</video_special>"
        self.IMAGE_START = "<image_special>"
        self.IMAGE_END = "</image_special>"
        self.NAV_TOKEN = "[Navigation]"
        self.IMAGE_SEP = "<image_sep>"

        # State for online inference
        self.rgb_list = []
        self.pending_actions = []

        print("Model loaded successfully")

    def reset(self):
        """Reset for new episode."""
        self.rgb_list = []
        self.pending_actions = []
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

    def add_frame(self, image: np.ndarray):
        """Add a new frame to history."""
        self.rgb_list.append(image)

    def process_images(self) -> List[torch.Tensor]:
        """Process accumulated images."""
        batch_image = np.asarray(self.rgb_list)
        self.model.get_model().new_frames = len(self.rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values']
        video = video.half().cuda()
        # Clear after processing (features are cached in model)
        self.rgb_list = []
        return [video]

    def predict(self, instruction: str) -> str:
        """Predict actions given accumulated frames."""
        prompt_template = (
            "This is a navigation video. The instruction is: {}\n"
            "Based on the visual observation and instruction, determine your next four actions. "
            "The predicted action should be one of the following: forward, left, right, or wait."
        )
        question = prompt_template.format(instruction)
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
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

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        imgs = self.process_images()

        with torch.inference_mode():
            self.model.update_prompt([[question]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

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
                actions.append('wait' if word == 'stop' else word)

        if len(actions) == 0:
            actions = ['wait'] * NUM_ACTIONS
        elif len(actions) < NUM_ACTIONS:
            while len(actions) < NUM_ACTIONS:
                actions.append(actions[-1])
        else:
            actions = actions[:NUM_ACTIONS]

        return actions

    def get_action(self, image: np.ndarray, instruction: str) -> Tuple[str, bool]:
        """
        Get next action given new frame.
        Returns (action, did_inference).

        Mimics agent_uninavid.py behavior:
        - Accumulate frames
        - If pending actions exist, return next pending action
        - Otherwise, run inference and get new actions
        """
        self.add_frame(image)

        # If we have pending actions, return next one
        if len(self.pending_actions) > 0:
            action = self.pending_actions.pop(0)
            return action, False

        # Run inference
        output = self.predict(instruction)
        actions = self.parse_actions(output)

        # Take first 2 actions (like agent_uninavid.py)
        self.pending_actions = actions[1:2]  # Keep 1 pending
        return actions[0], True


def evaluate_session_online(
    evaluator: OnlineActionEvaluator,
    session_path: str,
    stride: int = 60,
    verbose: bool = False
) -> Dict:
    """Evaluate a session with online inference."""
    # Load data
    with open(os.path.join(session_path, 'trajectory.json')) as f:
        trajectory = json.load(f)['trajectory']

    instruction = ""
    inst_file = os.path.join(session_path, 'instructions.json')
    if os.path.exists(inst_file):
        with open(inst_file) as f:
            inst_data = json.load(f)
        if 'instructions' in inst_data and 'description_overall' in inst_data['instructions']:
            instruction = inst_data['instructions']['description_overall']

    num_frames = len(trajectory)
    session_name = os.path.basename(session_path)

    # Reset evaluator
    evaluator.reset()

    # Simulate navigation
    results = []
    total_correct = 0
    total_actions = 0
    inference_count = 0

    # Process frames sequentially
    frame_idx = 0
    while frame_idx < num_frames:
        # Load current frame
        img_file = trajectory[frame_idx]['image_file']
        img_path = os.path.join(session_path, img_file)
        img = np.array(Image.open(img_path).convert('RGB'))

        # Get action
        pred_action, did_inference = evaluator.get_action(img, instruction)

        if did_inference:
            inference_count += 1

            # Calculate ground truth for this inference point
            current_pose = trajectory[frame_idx]['pose']
            future_idx = min(frame_idx + stride, num_frames - 1)
            future_pose = trajectory[future_idx]['pose']
            gt_actions = waypoint_to_actions(current_pose, future_pose)

            # Get all predicted actions for this inference
            all_pred = [pred_action] + evaluator.pending_actions.copy()
            while len(all_pred) < NUM_ACTIONS:
                all_pred.append(all_pred[-1])

            # Compare
            correct = sum(1 for p, g in zip(all_pred[:NUM_ACTIONS], gt_actions) if p == g)
            total_correct += correct
            total_actions += NUM_ACTIONS

            result = {
                'frame_idx': frame_idx,
                'gt_actions': gt_actions,
                'pred_actions': all_pred[:NUM_ACTIONS],
                'correct': correct,
            }
            results.append(result)

            if verbose:
                print(f"  Frame {frame_idx}: GT={' '.join(gt_actions)}, Pred={' '.join(all_pred[:NUM_ACTIONS])}, Correct={correct}/4")

        # Move to next frame (simulate action execution)
        frame_idx += 1

    accuracy = total_correct / total_actions if total_actions > 0 else 0

    return {
        'session': session_name,
        'num_frames': num_frames,
        'inference_count': inference_count,
        'total_correct': total_correct,
        'total_actions': total_actions,
        'accuracy': accuracy,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description='Online evaluation of VLN Action Text model')
    parser.add_argument('--model_path', type=str,
                        default='/mnt/dataset/wj_zqc/VLN/model/uninavid-7b-full-224-video-fps-1-grid-2')
    parser.add_argument('--lora_path', type=str,
                        default='output/vln_action_text_test')
    parser.add_argument('--data_path', type=str, default='passed_samples')
    parser.add_argument('--stride', type=int, default=60)
    parser.add_argument('--max_sessions', type=int, default=None)
    parser.add_argument('--output', type=str, default='eval_online_results.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = OnlineActionEvaluator(args.model_path, args.lora_path)

    # Find sessions
    session_dirs = []
    for item in sorted(os.listdir(args.data_path)):
        item_path = os.path.join(args.data_path, item)
        if os.path.isdir(item_path):
            traj_file = os.path.join(item_path, 'trajectory.json')
            if os.path.exists(traj_file):
                session_dirs.append(item_path)

    if args.max_sessions:
        session_dirs = session_dirs[:args.max_sessions]

    print(f"Evaluating {len(session_dirs)} sessions")
    print()

    # Evaluate each session
    all_results = []
    total_correct = 0
    total_actions = 0

    action_confusion = {a: Counter() for a in ['forward', 'left', 'right', 'wait']}

    for session_path in tqdm(session_dirs, desc="Sessions"):
        if args.verbose:
            print(f"\n{os.path.basename(session_path)}:")

        result = evaluate_session_online(
            evaluator, session_path, args.stride, args.verbose
        )
        all_results.append(result)

        total_correct += result['total_correct']
        total_actions += result['total_actions']

        # Update confusion matrix
        for r in result['results']:
            for gt, pred in zip(r['gt_actions'], r['pred_actions']):
                action_confusion[gt][pred] += 1

    # Calculate overall metrics
    overall_accuracy = total_correct / total_actions if total_actions > 0 else 0

    print("\n" + "=" * 50)
    print("ONLINE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total sessions: {len(session_dirs)}")
    print(f"Total inferences: {sum(r['inference_count'] for r in all_results)}")
    print(f"Action accuracy: {overall_accuracy:.2%} ({total_correct}/{total_actions})")

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

    # Per-session summary
    print("\nPer-session accuracy:")
    for r in all_results:
        print(f"  {r['session']}: {r['accuracy']:.2%} ({r['total_correct']}/{r['total_actions']}) - {r['inference_count']} inferences")

    # Save results
    output_data = {
        'config': {
            'model_path': args.model_path,
            'lora_path': args.lora_path,
            'stride': args.stride,
        },
        'metrics': {
            'total_sessions': len(session_dirs),
            'overall_accuracy': overall_accuracy,
        },
        'confusion_matrix': {gt: dict(action_confusion[gt]) for gt in actions},
        'sessions': all_results,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
