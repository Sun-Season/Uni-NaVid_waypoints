#!/usr/bin/env python3
"""
预处理 passed_samples，将 waypoint 转换为 action 并保存。
同时验证转换结果的正确性。
"""

import os
import json
import math
import argparse
from typing import List, Dict
from collections import Counter

# Waypoint to action conversion parameters
FORWARD_DISTANCE = 0.25  # meters per forward action
TURN_ANGLE = math.radians(30)  # 30 degrees per turn action
MIN_DISPLACEMENT = 0.05  # minimum displacement threshold
NUM_ACTIONS = 4  # fixed number of actions to predict
DEFAULT_STRIDE = 60  # default frame stride


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def waypoint_to_actions(current_pose: dict, future_pose: dict) -> List[str]:
    """Convert waypoint to discrete action sequence."""
    dx = future_pose['x'] - current_pose['x']
    dy = future_pose['y'] - current_pose['y']
    r = math.sqrt(dx*dx + dy*dy)
    yaw_diff = normalize_angle(future_pose['yaw'] - current_pose['yaw'])

    actions = []

    if r >= MIN_DISPLACEMENT:
        global_theta = math.atan2(dy, dx)
        relative_theta = normalize_angle(global_theta - current_pose['yaw'])

        # First turn towards target
        while abs(relative_theta) > TURN_ANGLE / 2:
            if relative_theta > 0:
                actions.append('left')
                relative_theta -= TURN_ANGLE
            else:
                actions.append('right')
                relative_theta += TURN_ANGLE

        # Then move forward
        remaining_dist = r
        while remaining_dist > FORWARD_DISTANCE / 2:
            actions.append('forward')
            remaining_dist -= FORWARD_DISTANCE
    else:
        # In-place rotation
        while abs(yaw_diff) > TURN_ANGLE / 2:
            if yaw_diff > 0:
                actions.append('left')
                yaw_diff -= TURN_ANGLE
            else:
                actions.append('right')
                yaw_diff += TURN_ANGLE

    # Handle action count
    raw_count = len(actions)
    if len(actions) == 0:
        actions = ['wait'] * NUM_ACTIONS
    else:
        if len(actions) > NUM_ACTIONS:
            actions = actions[:NUM_ACTIONS]
        while len(actions) < NUM_ACTIONS:
            actions.append(actions[-1])

    return actions, raw_count


def process_session(session_path: str, stride: int) -> Dict:
    """Process a single session and return action data."""
    traj_file = os.path.join(session_path, 'trajectory.json')

    with open(traj_file) as f:
        traj_data = json.load(f)

    trajectory = traj_data['trajectory']
    num_frames = len(trajectory)

    # Load instruction
    inst_file = os.path.join(session_path, 'instructions.json')
    instruction = ""
    if os.path.exists(inst_file):
        with open(inst_file) as f:
            inst_data = json.load(f)
        if 'instructions' in inst_data and 'description_overall' in inst_data['instructions']:
            instruction = inst_data['instructions']['description_overall']

    # Generate action samples
    samples = []
    for frame_idx in range(4, num_frames, stride):  # min_history=4
        future_idx = min(frame_idx + stride, num_frames - 1)

        current_pose = trajectory[frame_idx]['pose']
        future_pose = trajectory[future_idx]['pose']

        actions, raw_count = waypoint_to_actions(current_pose, future_pose)

        # Calculate displacement and angle for verification
        dx = future_pose['x'] - current_pose['x']
        dy = future_pose['y'] - current_pose['y']
        r = math.sqrt(dx*dx + dy*dy)
        yaw_diff = normalize_angle(future_pose['yaw'] - current_pose['yaw'])

        samples.append({
            'frame_idx': frame_idx,
            'future_idx': future_idx,
            'actions': actions,
            'action_str': ' '.join(actions),
            'raw_action_count': raw_count,
            'displacement': r,
            'yaw_change': math.degrees(yaw_diff),
            'current_pose': current_pose,
            'future_pose': future_pose,
        })

    return {
        'session_name': os.path.basename(session_path),
        'num_frames': num_frames,
        'instruction': instruction,
        'samples': samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocess passed_samples for action prediction')
    parser.add_argument('--data_path', type=str, default='passed_samples',
                        help='Path to passed_samples directory')
    parser.add_argument('--output', type=str, default='passed_samples_actions.json',
                        help='Output file for preprocessed actions')
    parser.add_argument('--stride', type=int, default=DEFAULT_STRIDE,
                        help='Frame stride for waypoint sampling')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    args = parser.parse_args()

    print(f"=== Preprocessing passed_samples ===")
    print(f"Data path: {args.data_path}")
    print(f"Stride: {args.stride}")
    print(f"Parameters: FORWARD={FORWARD_DISTANCE}m, TURN={math.degrees(TURN_ANGLE)}°")
    print()

    # Find all session directories
    session_dirs = []
    for item in sorted(os.listdir(args.data_path)):
        item_path = os.path.join(args.data_path, item)
        if os.path.isdir(item_path):
            traj_file = os.path.join(item_path, 'trajectory.json')
            if os.path.exists(traj_file):
                session_dirs.append(item_path)

    print(f"Found {len(session_dirs)} sessions")
    print()

    # Process all sessions
    all_sessions = []
    all_action_counts = Counter()
    total_samples = 0

    for session_path in session_dirs:
        session_data = process_session(session_path, args.stride)
        all_sessions.append(session_data)

        # Count actions
        for sample in session_data['samples']:
            for action in sample['actions']:
                all_action_counts[action] += 1
            total_samples += 1

        if args.verbose:
            print(f"{session_data['session_name']}: {len(session_data['samples'])} samples")
            for sample in session_data['samples'][:3]:
                print(f"  Frame {sample['frame_idx']}: {sample['action_str']} "
                      f"(r={sample['displacement']:.3f}m, Δyaw={sample['yaw_change']:.1f}°)")

    # Save to file
    output_data = {
        'metadata': {
            'stride': args.stride,
            'forward_distance': FORWARD_DISTANCE,
            'turn_angle': math.degrees(TURN_ANGLE),
            'num_actions': NUM_ACTIONS,
            'total_sessions': len(session_dirs),
            'total_samples': total_samples,
        },
        'action_distribution': dict(all_action_counts),
        'sessions': all_sessions,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"=== Results ===")
    print(f"Total sessions: {len(session_dirs)}")
    print(f"Total samples: {total_samples}")
    print()
    print("Action distribution:")
    total_actions = sum(all_action_counts.values())
    for action in ['forward', 'left', 'right', 'wait']:
        count = all_action_counts[action]
        print(f"  {action}: {count} ({count/total_actions*100:.1f}%)")
    print()
    print(f"Saved to: {args.output}")

    # Verification: check some samples
    print()
    print("=== Verification ===")

    # Check displacement vs forward count
    print("Checking displacement vs forward actions:")
    errors = []
    for session in all_sessions:
        for sample in session['samples']:
            r = sample['displacement']
            forward_count = sample['actions'].count('forward')
            expected_forwards = round(r / FORWARD_DISTANCE) if r >= MIN_DISPLACEMENT else 0

            # Allow some tolerance due to truncation/padding
            if abs(forward_count - expected_forwards) > 2 and sample['raw_action_count'] <= NUM_ACTIONS:
                errors.append({
                    'session': session['session_name'],
                    'frame': sample['frame_idx'],
                    'displacement': r,
                    'expected': expected_forwards,
                    'actual': forward_count,
                })

    if errors:
        print(f"  Found {len(errors)} potential issues:")
        for err in errors[:5]:
            print(f"    {err['session']} frame {err['frame']}: "
                  f"r={err['displacement']:.3f}m, expected ~{err['expected']} forwards, got {err['actual']}")
    else:
        print("  All samples passed displacement check!")

    # Show sample distribution
    print()
    print("Sample action patterns (top 10):")
    pattern_counts = Counter()
    for session in all_sessions:
        for sample in session['samples']:
            pattern_counts[sample['action_str']] += 1

    for pattern, count in pattern_counts.most_common(10):
        print(f"  '{pattern}': {count} ({count/total_samples*100:.1f}%)")


if __name__ == '__main__':
    main()
