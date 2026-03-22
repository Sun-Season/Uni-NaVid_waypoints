#!/usr/bin/env python3
"""
设置动作token的训练权重
获取动作词的token ID并创建权重字典
"""

import os
import sys
import json
from collections import defaultdict
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_action_token_ids(tokenizer, action_words):
    """获取动作词的候选 token ID。"""

    print(f"{'='*60}")
    print("获取动作词的Token ID")
    print(f"{'='*60}\n")

    action_token_ids = {}

    for action in action_words:
        # 尝试不同的tokenization方式
        # 方式1: 直接tokenize
        tokens_direct = tokenizer.encode(action, add_special_tokens=False)

        # 方式2: 带空格前缀（因为在句子中间的词通常有空格）
        tokens_with_space = tokenizer.encode(f" {action}", add_special_tokens=False)

        # 方式3: 首字母大写
        tokens_capitalized = tokenizer.encode(action.capitalize(), add_special_tokens=False)

        print(f"动作: {action}")
        print(f"  直接tokenize: {tokens_direct} -> {tokenizer.convert_ids_to_tokens(tokens_direct)}")
        print(f"  带空格: {tokens_with_space} -> {tokenizer.convert_ids_to_tokens(tokens_with_space)}")
        print(f"  首字母大写: {tokens_capitalized} -> {tokenizer.convert_ids_to_tokens(tokens_capitalized)}")

        # 存储所有可能的 token ID，用于后续过滤共享 token。
        all_token_ids = sorted(set(tokens_direct + tokens_with_space + tokens_capitalized))
        action_token_ids[action] = all_token_ids
        print(f"  所有候选 token IDs: {all_token_ids}\n")

    return action_token_ids


def split_shared_and_unique_tokens(action_token_ids):
    """区分共享 token 和动作独有 token。"""

    token_to_actions = defaultdict(list)
    for action, token_ids in action_token_ids.items():
        for token_id in token_ids:
            token_to_actions[token_id].append(action)

    shared_token_ids = {
        token_id: sorted(actions)
        for token_id, actions in token_to_actions.items()
        if len(actions) > 1
    }
    weighted_action_token_ids = {
        action: [token_id for token_id in token_ids if token_id not in shared_token_ids]
        for action, token_ids in action_token_ids.items()
    }

    return weighted_action_token_ids, shared_token_ids


def create_weight_dict(weighted_action_token_ids, action_weights, shared_token_ids):
    """创建 token ID 到权重的映射，跳过共享 token。"""

    print(f"{'='*60}")
    print("创建Token权重字典")
    print(f"{'='*60}\n")

    token_weight_dict = {}

    if shared_token_ids:
        print("跳过共享 token（通常是空格/前缀 token）:")
        for token_id, actions in sorted(shared_token_ids.items()):
            print(f"  Token ID {token_id:5d} shared by {', '.join(actions)}")
        print()

    for action, token_ids in weighted_action_token_ids.items():
        weight = action_weights.get(action, 1.0)
        if not token_ids:
            print(f"动作 {action:8s} 没有独有 token，将保持默认权重 1.0")
            continue
        for token_id in token_ids:
            token_weight_dict[token_id] = weight
            print(f"Token ID {token_id:5d} ({action:8s}) -> weight {weight:.4f}")

    print(f"\n总共设置了 {len(token_weight_dict)} 个独有 token 的权重\n")

    return token_weight_dict


def main():
    # 模型路径
    model_path = "/mnt/dataset/shuhzeng/model/uninavid-7b-full-224-video-fps-1-grid-2"

    print(f"加载tokenizer: {model_path}\n")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 动作词列表
    action_words = ['forward', 'left', 'right', 'wait', 'stop']

    # 推荐的权重（基于任务重要性调整）
    action_weights = {
        'forward': 1.0000,
        'left': 1.5000,    # 适度提高转弯权重
        'right': 1.5000,   # 适度提高转弯权重
        'wait': 1.1000,    # 轻微提高等待权重，避免过度保守
        'stop': 1.2500,    # stop 比 wait 更关键，略高一些
    }

    print("推荐的动作权重:")
    for action, weight in action_weights.items():
        print(f"  {action:8s}: {weight:.4f}")
    print()

    # 获取token IDs
    action_token_ids = get_action_token_ids(tokenizer, action_words)

    weighted_action_token_ids, shared_token_ids = split_shared_and_unique_tokens(action_token_ids)

    print(f"{'='*60}")
    print("共享 / 独有 Token 分析")
    print(f"{'='*60}\n")
    for action in action_words:
        print(
            f"{action:8s} 候选 {action_token_ids[action]} "
            f"-> 加权 {weighted_action_token_ids[action]}"
        )
    if shared_token_ids:
        print("\n共享 token 将不参与加权:")
        for token_id, actions in sorted(shared_token_ids.items()):
            print(f"  {token_id}: {actions}")
    print()

    # 创建权重字典
    token_weight_dict = create_weight_dict(weighted_action_token_ids, action_weights, shared_token_ids)

    # 保存配置
    output = {
        'action_weights': action_weights,
        'action_token_ids': action_token_ids,
        'weighted_action_token_ids': weighted_action_token_ids,
        'shared_token_ids': shared_token_ids,
        'token_weight_dict': token_weight_dict,
        'vocab_size': tokenizer.vocab_size,
    }

    output_file = 'action_token_weights.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"{'='*60}")
    print("配置已保存")
    print(f"{'='*60}\n")
    print(f"权重配置已保存到: {output_file}\n")

    # 生成训练脚本中使用的代码
    print(f"{'='*60}")
    print("在训练脚本中使用这些权重")
    print(f"{'='*60}\n")

    print("在训练脚本中添加以下代码：\n")
    print("```python")
    print("# 加载动作token权重")
    print("import json")
    print(f"with open('{output_file}', 'r') as f:")
    print("    weight_config = json.load(f)")
    print("token_weight_dict = {int(k): v for k, v in weight_config['token_weight_dict'].items()}")
    print()
    print("# 设置到模型config")
    print("model.config.action_token_weights = token_weight_dict")
    print("```\n")

    print("或者在模型加载时设置：\n")
    print("```python")
    print("from transformers import AutoConfig")
    print("config = AutoConfig.from_pretrained(model_path)")
    print("config.action_token_weights = token_weight_dict")
    print("model = AutoModelForCausalLM.from_pretrained(model_path, config=config)")
    print("```")


if __name__ == '__main__':
    main()
