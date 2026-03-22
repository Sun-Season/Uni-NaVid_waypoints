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

动作生成逻辑:
- Wait: 相邻 waypoint 时间间隔 > 1秒
- Turn: 累积 yaw 变化达到 ±20° 生成 left/right
- Forward: 累积位移达到 0.5m 生成 forward（减少forward主导）
- Stop: 轨迹结束

特性:
- 无重叠采样：窗口大小=4，步长=4，每个样本包含4个独立的动作
- 样本之间完全独立，无时序混乱问题
"""

import os
import json
import math
import argparse
from typing import List, Dict, Tuple
from collections import Counter
from pathlib import Path

# Waypoint to action conversion parameters
FORWARD_DISTANCE = 0.5  # meters per forward action (changed from 0.25 to reduce forward dominance)
TURN_ANGLE = 20.0  # degrees per turn action
WAIT_TIME_THRESHOLD = 2.0  # seconds, time gap > this triggers wait
WAIT_TIME_PER_ACTION = 2.0  # seconds per wait action
STATIONARY_THRESHOLD = 0.03  # meters, if displacement < this, consider as stationary
VIDEO_FPS = 30  # video frame rate

# Sampling parameters (NO OVERLAP)
WINDOW_SIZE = 4  # number of actions per sample
WINDOW_STRIDE = 4  # step size for sampling (equal to window_size = no overlap)

MAX_SINGLE_STEP_DISTANCE = 2.0  # max distance between adjacent waypoints (meters)


def normalize_angle_deg(angle: float) -> float:
    """Normalize angle to [-180, 180] degrees."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def get_video_frame(time_s: float) -> int:
    """Convert time_s to video frame number."""
    return int(time_s * VIDEO_FPS)


def generate_full_action_sequence(waypoints: List[dict], units_in_meters: float) -> List[dict]:
    """Generate complete action sequence from all waypoints.

    Logic:
    1. Wait: 当robot静止（位移<0.01m）时累积时间，每累积2秒生成一个wait
    2. Turn: 累积 yaw 变化达到 ±20° → 生成 left/right（先处理）
    3. Forward: 累积位移达到 0.25m → 生成 forward（后处理）
    4. Stop: 轨迹结束时生成

    Returns:
        List of action entries with action, wp_idx, time_s, video_frame
    """
    if len(waypoints) < 2:
        return []

    action_sequence = []

    # Accumulators
    accumulated_dist = 0.0
    accumulated_yaw = 0.0  # in degrees
    accumulated_wait_time = 0.0  # for stationary detection
    wait_start_wp_idx = None  # track where wait started

    for i in range(1, len(waypoints)):
        prev_wp = waypoints[i - 1]
        curr_wp = waypoints[i]

        time_gap = curr_wp['time_s'] - prev_wp['time_s']

        # Calculate displacement
        dx = (curr_wp['xyz'][0] - prev_wp['xyz'][0]) * units_in_meters
        dy = (curr_wp['xyz'][1] - prev_wp['xyz'][1]) * units_in_meters
        step_dist = math.sqrt(dx*dx + dy*dy)

        # Check for anomalous jump (teleport)
        if step_dist > MAX_SINGLE_STEP_DISTANCE:
            accumulated_dist = 0.0
            accumulated_yaw = 0.0
            accumulated_wait_time = 0.0
            wait_start_wp_idx = None
            continue

        # 1. Check for stationary (wait): displacement < threshold
        if step_dist < STATIONARY_THRESHOLD:
            # Robot is stationary, accumulate wait time
            if wait_start_wp_idx is None:
                wait_start_wp_idx = i - 1
            accumulated_wait_time += time_gap

            # Generate wait actions when accumulated time >= threshold
            while accumulated_wait_time >= WAIT_TIME_PER_ACTION:
                action_sequence.append({
                    'action': 'wait',
                    'wp_idx': wait_start_wp_idx,
                    'future_wp_idx': i,
                    'time_s': waypoints[wait_start_wp_idx]['time_s'],
                    'video_frame': get_video_frame(waypoints[wait_start_wp_idx]['time_s']),
                })
                accumulated_wait_time -= WAIT_TIME_PER_ACTION
            continue

        # Robot is moving, reset wait accumulator
        accumulated_wait_time = 0.0
        wait_start_wp_idx = None

        # 2. Accumulate yaw change (in degrees)
        prev_yaw = prev_wp['yaw_deg']
        curr_yaw = curr_wp['yaw_deg']
        yaw_change = normalize_angle_deg(curr_yaw - prev_yaw)
        accumulated_yaw += yaw_change

        # Generate turn actions (process turn BEFORE forward)
        while abs(accumulated_yaw) >= TURN_ANGLE:
            if accumulated_yaw > 0:
                action_sequence.append({
                    'action': 'left',
                    'wp_idx': i - 1,
                    'future_wp_idx': i,
                    'time_s': prev_wp['time_s'],
                    'video_frame': get_video_frame(prev_wp['time_s']),
                })
                accumulated_yaw -= TURN_ANGLE
            else:
                action_sequence.append({
                    'action': 'right',
                    'wp_idx': i - 1,
                    'future_wp_idx': i,
                    'time_s': prev_wp['time_s'],
                    'video_frame': get_video_frame(prev_wp['time_s']),
                })
                accumulated_yaw += TURN_ANGLE

        # 3. Accumulate displacement
        accumulated_dist += step_dist

        # Generate forward actions (process forward AFTER turn)
        while accumulated_dist >= FORWARD_DISTANCE:
            action_sequence.append({
                'action': 'forward',
                'wp_idx': i - 1,
                'future_wp_idx': i,
                'time_s': prev_wp['time_s'],
                'video_frame': get_video_frame(prev_wp['time_s']),
            })
            accumulated_dist -= FORWARD_DISTANCE

    # 4. Add final 'stop' actions (4 stops to ensure final window is all stops)
    last_wp = waypoints[-1]
    for _ in range(WINDOW_SIZE):
        action_sequence.append({
            'action': 'stop',
            'wp_idx': len(waypoints) - 1,
            'future_wp_idx': len(waypoints) - 1,
            'time_s': last_wp['time_s'],
            'video_frame': get_video_frame(last_wp['time_s']),
        })

    return action_sequence


def sliding_window_samples(action_sequence: List[dict],
                           window_size: int = WINDOW_SIZE,
                           stride: int = WINDOW_STRIDE) -> List[dict]:
    """Apply non-overlapping sampling to action sequence to create samples.

    Args:
        action_sequence: Full action sequence with metadata
        window_size: Number of actions per sample (default 4)
        stride: Step size for sampling (default 4, equal to window_size for no overlap)

    Returns:
        List of samples, each with 'actions' and metadata
    """
    samples = []

    if len(action_sequence) < window_size:
        # Not enough actions, pad with wait
        actions = [a['action'] for a in action_sequence]
        while len(actions) < window_size:
            actions.append('wait')

        first_entry = action_sequence[0] if action_sequence else None
        if first_entry:
            samples.append({
                'wp_idx': first_entry['wp_idx'],
                'video_frame': first_entry['video_frame'],
                'time_s': first_entry['time_s'],
                'actions': actions,
                'action_str': ' '.join(actions),
            })
        return samples

    # Non-overlapping sampling - do NOT handle remaining actions
    # to avoid overlap. It's acceptable to not cover the last few actions.
    for i in range(0, len(action_sequence) - window_size + 1, stride):
        window = action_sequence[i:i + window_size]
        actions = [a['action'] for a in window]

        # Use first action's metadata for the sample
        first_entry = window[0]
        last_entry = window[-1]

        samples.append({
            'wp_idx': first_entry['wp_idx'],
            'future_wp_idx': last_entry['future_wp_idx'],
            'video_frame': first_entry['video_frame'],
            'future_video_frame': last_entry['video_frame'],
            'time_s': first_entry['time_s'],
            'future_time_s': last_entry['time_s'],
            'actions': actions,
            'action_str': ' '.join(actions),
        })

    return samples


def process_episode(json_path: str, window_size: int = WINDOW_SIZE,
                    stride: int = WINDOW_STRIDE) -> Dict:
    """Process a single episode and return action data.

    Args:
        json_path: Path to episode JSON file
        window_size: Sliding window size (default 4)
        stride: Sliding window stride (default 2)
    """
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

    # Step 1: Generate full action sequence from all waypoints
    action_sequence = generate_full_action_sequence(waypoints, units_in_meters)

    if len(action_sequence) == 0:
        return None

    # Step 2: Apply sliding window to create samples
    samples = sliding_window_samples(action_sequence, window_size, stride)

    return {
        'scene_id': scene_id,
        'instruction': instruction,
        'num_waypoints': len(waypoints),
        'total_actions': len(action_sequence),
        'window_size': window_size,
        'window_stride': stride,
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
                        default='/mnt/dataset/shuzheng/OmniNavBench/OmniNavBenchData',
                        help='Path to OmniNavBenchData directory')
    parser.add_argument('--output_root', type=str,
                        default='/mnt/dataset/shuzheng/OmniNavBench/OmniNavBenchActionData',
                        help='Path to output OmniNavBenchActionData directory')
    parser.add_argument('--agent_types', type=str, nargs='+', default=None,
                        help='Agent types to process (default: car dog human)')
    parser.add_argument('--inst_types', type=str, nargs='+', default=None,
                        help='Instruction types to process (default: original concise first_person verbose)')
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE,
                        help='Sliding window size (default: 4)')
    parser.add_argument('--window_stride', type=int, default=WINDOW_STRIDE,
                        help='Sliding window stride (default: 2)')
    parser.add_argument('--split', type=str, default=None,
                        help='Only process specific split (train/test)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of episodes to process')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    parser.add_argument('--dry_run', action='store_true',
                        help='Do not write files, just show what would be done')
    args = parser.parse_args()

    print(f"=== Preprocessing OmniNavBench (Sliding Window) ===")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Window size: {args.window_size}, Stride: {args.window_stride}")
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
    all_pattern_counts = Counter()
    total_samples = 0
    success_count = 0
    skip_count = 0

    for i, ep_info in enumerate(episodes):
        if args.verbose or (i + 1) % 100 == 0:
            print(f"Processing {i+1}/{len(episodes)}: {ep_info['inst_type']}/{ep_info['agent_type']}/{ep_info['scene']}/{ep_info['episode']}")

        try:
            episode_data = process_episode(ep_info['json_path'], args.window_size, args.window_stride)

            if episode_data is None or len(episode_data['samples']) == 0:
                skip_count += 1
                continue

            # Add metadata
            episode_data['split'] = ep_info['split']
            episode_data['inst_type'] = ep_info['inst_type']
            episode_data['agent_type'] = ep_info['agent_type']
            episode_data['episode_id'] = ep_info['episode']

            # Build output path
            output_dir = Path(args.output_root) / ep_info['split'] / ep_info['inst_type'] / ep_info['agent_type'] / ep_info['scene'] / ep_info['episode']
            output_file = output_dir / 'actions.json'

            # Create directory and write file
            if not args.dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(episode_data, f, indent=2)

            success_count += 1

            # Count actions and patterns
            for sample in episode_data['samples']:
                for action in sample['actions']:
                    all_action_counts[action] += 1
                all_pattern_counts[tuple(sample['actions'])] += 1
                total_samples += 1

            if args.verbose:
                print(f"  -> {output_file}")
                print(f"     {len(episode_data['samples'])} samples from {episode_data['total_actions']} actions")

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

    print()
    print("Top 15 action patterns:")
    for pattern, count in all_pattern_counts.most_common(15):
        pct = count / total_samples * 100 if total_samples > 0 else 0
        print(f"  {list(pattern)}: {count} ({pct:.1f}%)")

    # Count same vs mixed patterns
    same_count = sum(1 for p in all_pattern_counts if len(set(p)) == 1)
    mixed_count = len(all_pattern_counts) - same_count
    same_samples = sum(c for p, c in all_pattern_counts.items() if len(set(p)) == 1)
    mixed_samples = total_samples - same_samples
    print()
    print(f"Pattern diversity:")
    print(f"  All same actions: {same_samples} samples ({same_samples/total_samples*100:.1f}%)")
    print(f"  Mixed actions: {mixed_samples} samples ({mixed_samples/total_samples*100:.1f}%)")

    if args.dry_run:
        print()
        print("(Dry run - no files were written)")
    else:
        print()
        print(f"Output saved to: {args.output_root}")


if __name__ == '__main__':
    main()
