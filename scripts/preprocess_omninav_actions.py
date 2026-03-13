#!/usr/bin/env python3
"""
预处理 OmniNavBench 数据，将 waypoint 转换为离散 action 并保存。

输出到 OmniNavBenchActionData 目录，结构与 OmniNavBenchVideos 一致：
  OmniNavBenchActionData/
    train/
      car/
        scene_id/
          episode_id/
            actions.json

对齐方式: video_frame = int(time_s * 30)
参数: FORWARD_DISTANCE=0.25m, TURN_ANGLE=30°

特性:
- 智能采样：直线段稀疏采样，转弯段密集采样
- STOP action：每个 episode 最后一个 sample 的 actions 为 ['stop', 'stop', 'stop', 'stop']
"""

import os
import json
import math
import argparse
from typing import List, Dict, Tuple
from collections import Counter
from pathlib import Path

# Waypoint to action conversion parameters
FORWARD_DISTANCE = 0.25  # meters per forward action
TURN_ANGLE = math.radians(30)  # 30 degrees per turn action
MIN_DISPLACEMENT = 0.05  # minimum displacement threshold
NUM_ACTIONS = 4  # fixed number of actions to predict
VIDEO_FPS = 30  # video frame rate

# Smart sampling parameters
TURN_DETECTION_THRESHOLD = 0.1  # radians (~6 degrees)
STRAIGHT_SAMPLE_INTERVAL = 5  # sample every 5 waypoints in straight segments
TURN_SAMPLE_INTERVAL = 1  # sample every waypoint in turning segments
USE_SMART_SAMPLING = True  # enable/disable smart sampling


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def waypoint_to_actions(current_pose: dict, future_pose: dict) -> Tuple[List[str], int]:
    """Convert waypoint to discrete action sequence.

    Args:
        current_pose: dict with 'x', 'y', 'yaw' (yaw in radians)
        future_pose: dict with 'x', 'y', 'yaw' (yaw in radians)

    Returns:
        (actions, raw_count) tuple
    """
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


def omninav_wp_to_pose(wp: dict, units_in_meters: float = 1.0) -> dict:
    """Convert OmniNavBench waypoint to pose dict."""
    return {
        'x': wp['xyz'][0] * units_in_meters,
        'y': wp['xyz'][1] * units_in_meters,
        'yaw': math.radians(wp['yaw_deg'])
    }


def get_video_frame(time_s: float) -> int:
    """Convert time_s to video frame number."""
    return int(time_s * VIDEO_FPS)


MAX_SINGLE_STEP_DISTANCE = 2.0  # max distance between adjacent waypoints (meters)


def detect_trajectory_features(waypoints: List[dict], units_in_meters: float = 1.0) -> List[str]:
    """Detect trajectory features: straight segments vs turning segments."""
    if len(waypoints) < 3:
        return ['start'] + ['straight'] * (len(waypoints) - 2) + ['end'] if len(waypoints) > 1 else ['start']

    features = []

    for i in range(len(waypoints)):
        if i == 0:
            features.append('start')
        elif i == len(waypoints) - 1:
            features.append('end')
        else:
            prev_yaw = math.radians(waypoints[i - 1]['yaw_deg'])
            next_yaw = math.radians(waypoints[i + 1]['yaw_deg'])
            angle_change = abs(normalize_angle(next_yaw - prev_yaw))

            if angle_change > TURN_DETECTION_THRESHOLD:
                features.append('turning')
            else:
                features.append('straight')

    return features


def smart_sample_waypoints(waypoints: List[dict], features: List[str]) -> List[int]:
    """Intelligent waypoint sampling: sparse for straight, dense for turning."""
    if not USE_SMART_SAMPLING or len(waypoints) < 3:
        return list(range(len(waypoints)))

    sampled_indices = [0]  # Always include start

    i = 1
    while i < len(waypoints) - 1:
        feature = features[i]

        if feature == 'turning':
            interval = TURN_SAMPLE_INTERVAL
        else:  # straight
            interval = STRAIGHT_SAMPLE_INTERVAL

        if i % interval == 0 or feature == 'turning':
            sampled_indices.append(i)

        i += 1

    # Always include end
    if sampled_indices[-1] != len(waypoints) - 1:
        sampled_indices.append(len(waypoints) - 1)

    return sampled_indices


def process_episode(json_path: str, stride_distance: float = 0.25, use_smart_sampling: bool = True) -> Dict:
    """Process a single episode and return action data."""
    with open(json_path) as f:
        data = json.load(f)

    scenario = data['scenarios'][0]
    scene_id = scenario['id']
    instruction = scenario['task']['navigation']['instruction']
    waypoints = scenario['robots']['entries'][0]['rb_gt_waypoints']

    # Get units_in_meters from scene config (default 1.0 = meters)
    units_in_meters = scenario['scene'].get('units_in_meters', 1.0)

    if len(waypoints) < 2:
        return None

    # Detect trajectory features for smart sampling
    features = detect_trajectory_features(waypoints, units_in_meters)

    # Get sampled waypoint indices
    if use_smart_sampling and USE_SMART_SAMPLING:
        sampled_indices = smart_sample_waypoints(waypoints, features)
    else:
        sampled_indices = list(range(len(waypoints)))

    # Generate action samples from sampled waypoints
    samples = []

    for i in range(len(sampled_indices) - 1):
        wp_idx = sampled_indices[i]
        future_idx = sampled_indices[i + 1]

        current_wp = waypoints[wp_idx]
        future_wp = waypoints[future_idx]

        # Check for anomalous distance
        dx = (future_wp['xyz'][0] - current_wp['xyz'][0]) * units_in_meters
        dy = (future_wp['xyz'][1] - current_wp['xyz'][1]) * units_in_meters
        dist = math.sqrt(dx*dx + dy*dy)

        if dist > MAX_SINGLE_STEP_DISTANCE:
            continue  # Skip anomalous waypoints

        # Convert to pose (with unit conversion)
        current_pose = omninav_wp_to_pose(current_wp, units_in_meters)
        future_pose = omninav_wp_to_pose(future_wp, units_in_meters)

        # Generate actions
        actions, raw_count = waypoint_to_actions(current_pose, future_pose)

        # Calculate displacement and angle for verification
        r = dist
        yaw_diff = normalize_angle(future_pose['yaw'] - current_pose['yaw'])

        # Video frame alignment
        current_video_frame = get_video_frame(current_wp['time_s'])
        future_video_frame = get_video_frame(future_wp['time_s'])

        samples.append({
            'wp_idx': wp_idx,
            'future_wp_idx': future_idx,
            'video_frame': current_video_frame,
            'future_video_frame': future_video_frame,
            'time_s': current_wp['time_s'],
            'future_time_s': future_wp['time_s'],
            'actions': actions,
            'action_str': ' '.join(actions),
            'raw_action_count': raw_count,
            'displacement': r,
            'yaw_change': math.degrees(yaw_diff),
            'current_pose': current_pose,
            'future_pose': future_pose,
            'feature': features[wp_idx],
        })

    # Set last sample's actions to STOP
    if len(samples) > 0:
        samples[-1]['actions'] = ['stop'] * NUM_ACTIONS
        samples[-1]['action_str'] = ' '.join(samples[-1]['actions'])

    return {
        'scene_id': scene_id,
        'instruction': instruction,
        'num_waypoints': len(waypoints),
        'num_sampled': len(sampled_indices),
        'samples': samples,
    }


def find_all_episodes(data_root: str, agent_types: List[str] = None,
                      instruction_types: List[str] = None) -> List[Dict]:
    """Find all episode JSON files."""
    data_root = Path(data_root)

    if agent_types is None:
        agent_types = ['car', 'dog', 'human']

    if instruction_types is None:
        instruction_types = ['original', 'concise', 'first_person', 'verbose']

    episodes = []

    # Check both train and test splits
    for split in ['train', 'test']:
        for inst_type in instruction_types:
            split_dir = data_root / split / inst_type
            if not split_dir.exists():
                continue

            for agent_type in agent_types:
                agent_dir = split_dir / agent_type
                if not agent_dir.exists():
                    continue

                # Iterate through scenes
                for scene_dir in sorted(agent_dir.iterdir()):
                    if not scene_dir.is_dir():
                        continue

                    # Find all episode JSON files
                    for json_file in sorted(scene_dir.glob('final_episode_*.json')):
                        episodes.append({
                            'json_path': str(json_file),
                            'split': split,
                            'inst_type': inst_type,
                            'agent_type': agent_type,
                            'scene': scene_dir.name,
                            'episode': json_file.stem,
                        })

    return episodes


def main():
    parser = argparse.ArgumentParser(description='Preprocess OmniNavBench for action prediction')
    parser.add_argument('--data_root', type=str,
                        default='/mnt/dataset/shuhzeng/OmniNavBench/OmniNavBenchData',
                        help='Path to OmniNavBenchData directory')
    parser.add_argument('--output_root', type=str,
                        default='/mnt/dataset/shuhzeng/OmniNavBench/OmniNavBenchActionData',
                        help='Path to output OmniNavBenchActionData directory')
    parser.add_argument('--agent_types', type=str, nargs='+', default=None,
                        help='Agent types to process (default: car dog human)')
    parser.add_argument('--inst_types', type=str, nargs='+', default=None,
                        help='Instruction types to process (default: original concise first_person verbose)')
    parser.add_argument('--stride_distance', type=float, default=0.25,
                        help='Target distance between samples (meters)')
    parser.add_argument('--split', type=str, default=None,
                        help='Only process specific split (train/test)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of episodes to process')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    parser.add_argument('--dry_run', action='store_true',
                        help='Do not write files, just show what would be done')
    args = parser.parse_args()

    print(f"=== Preprocessing OmniNavBench ===")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Stride distance: {args.stride_distance}m")
    print(f"Parameters: FORWARD={FORWARD_DISTANCE}m, TURN={math.degrees(TURN_ANGLE)}°")
    print()

    # Find all episodes
    episodes = find_all_episodes(args.data_root, args.agent_types, args.inst_types)

    if args.split:
        episodes = [e for e in episodes if e['split'] == args.split]

    if args.limit:
        episodes = episodes[:args.limit]

    print(f"Found {len(episodes)} episodes")
    print()

    # Process all episodes
    all_action_counts = Counter()
    total_samples = 0
    success_count = 0
    skip_count = 0

    for i, ep_info in enumerate(episodes):
        if args.verbose or (i + 1) % 100 == 0:
            print(f"Processing {i+1}/{len(episodes)}: {ep_info['inst_type']}/{ep_info['agent_type']}/{ep_info['scene']}/{ep_info['episode']}")

        try:
            episode_data = process_episode(ep_info['json_path'], args.stride_distance)

            if episode_data is None or len(episode_data['samples']) == 0:
                skip_count += 1
                continue

            # Add metadata
            episode_data['split'] = ep_info['split']
            episode_data['inst_type'] = ep_info['inst_type']
            episode_data['agent_type'] = ep_info['agent_type']
            episode_data['episode_id'] = ep_info['episode']

            # Build output path: output_root/split/inst_type/agent_type/scene/episode/actions.json
            output_dir = Path(args.output_root) / ep_info['split'] / ep_info['inst_type'] / ep_info['agent_type'] / ep_info['scene'] / ep_info['episode']
            output_file = output_dir / 'actions.json'

            # Create directory and write file
            if not args.dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(episode_data, f, indent=2)

            success_count += 1

            # Count actions
            for sample in episode_data['samples']:
                for action in sample['actions']:
                    all_action_counts[action] += 1
                total_samples += 1

            if args.verbose:
                print(f"  -> {output_file}")
                print(f"     {len(episode_data['samples'])} samples")

        except Exception as e:
            print(f"  Error: {e}")
            skip_count += 1
            continue

    # Print summary
    print()
    print(f"=== Results ===")
    print(f"Successfully processed: {success_count} episodes")
    print(f"Skipped: {skip_count} episodes")
    print(f"Total samples: {total_samples}")
    print()
    print("Action distribution:")
    total_actions = sum(all_action_counts.values())
    for action in ['forward', 'left', 'right', 'wait', 'stop']:
        count = all_action_counts.get(action, 0)
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {action}: {count} ({pct:.1f}%)")

    if args.dry_run:
        print()
        print("(Dry run - no files were written)")
    else:
        print()
        print(f"Output saved to: {args.output_root}")


if __name__ == '__main__':
    main()
