#!/usr/bin/env python3
"""
检查训练数据中的动作分布
"""

import os
import sys
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uninavid.train.omninav_action_dataset import OmniNavActionDataset, OmniNavActionDataArguments


def analyze_action_distribution(data_root: str, split: str = 'train'):
    """分析指定split的动作分布"""

    print(f"{'='*60}")
    print(f"分析{split}集的动作分布")
    print(f"{'='*60}\n")

    # 创建数据集参数（不需要image_processor和tokenizer来统计）
    data_args = OmniNavActionDataArguments(
        action_root=data_root,
        video_root='/mnt/dataset/shuhzeng/OmniNavBench/OmniNavBenchVideos',
        split=split,
        enable_oversampling=False,
    )

    # 加载数据集（传入None作为tokenizer，因为我们只需要统计）
    dataset = OmniNavActionDataset(
        tokenizer=None,
        data_args=data_args
    )

    print(f"总样本数: {len(dataset.samples)}\n")

    # 统计动作分布
    action_counter = Counter()

    for ep_key, sample_idx, sample_data in dataset.samples:
        actions = sample_data.get('actions', [])
        for action in actions:
            action_counter[action] += 1

    # 输出结果
    total_actions = sum(action_counter.values())

    print(f"总动作数: {total_actions}\n")
    print("动作分布:")
    print(f"{'动作':<10} {'数量':>10} {'占比':>10}")
    print("-" * 32)

    for action in ['forward', 'left', 'right', 'wait', 'stop']:
        count = action_counter.get(action, 0)
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"{action:<10} {count:>10} {pct:>9.2f}%")

    print("\n" + "="*60)

    # 分析
    forward_pct = action_counter.get('forward', 0) / total_actions * 100
    left_pct = action_counter.get('left', 0) / total_actions * 100
    right_pct = action_counter.get('right', 0) / total_actions * 100
    wait_pct = action_counter.get('wait', 0) / total_actions * 100

    print("\n分析:")
    if forward_pct > 70:
        print(f"⚠️  forward占比过高 ({forward_pct:.1f}%)，存在严重的类别不平衡")
    elif forward_pct > 60:
        print(f"⚠️  forward占比较高 ({forward_pct:.1f}%)，存在类别不平衡")
    else:
        print(f"✓ forward占比合理 ({forward_pct:.1f}%)")

    if left_pct < 5:
        print(f"⚠️  left占比过低 ({left_pct:.1f}%)，可能导致模型难以学习")

    if right_pct < 5:
        print(f"⚠️  right占比过低 ({right_pct:.1f}%)，可能导致模型难以学习")

    return action_counter


if __name__ == '__main__':
    data_root = '/mnt/dataset/shuhzeng/OmniNavBench/OmniNavBenchActionData'

    # 分析训练集
    train_counter = analyze_action_distribution(data_root, split='train')

    print("\n\n")

    # 分析验证集
    val_counter = analyze_action_distribution(data_root, split='val')
