#!/usr/bin/env python3
"""
验证 action token 权重是否正确设置。

Usage:
    python scripts/verify_action_weights.py
"""

import sys
sys.path.insert(0, '/mnt/dataset/shuhzeng/Uni-NaVid_waypoints')

import torch
from transformers import AutoTokenizer
from uninavid.train.action_token_weights import (
    ACTION_LABELS,
    ACTION_TOKEN_WEIGHTS,
    get_action_token_ids,
    build_token_weight_tensor,
    print_action_token_info,
)


def main():
    # 加载 tokenizer
    model_path = "/mnt/dataset/shuzheng/model/uninavid-7b-full-224-video-fps-1-grid-2"
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    print("\n" + "=" * 60)
    print("1. Action Token IDs")
    print("=" * 60)
    print_action_token_info(tokenizer)

    print("\n" + "=" * 60)
    print("2. Weight Tensor Verification")
    print("=" * 60)

    vocab_size = 32000
    weights = build_token_weight_tensor(tokenizer, vocab_size=vocab_size)

    # 验证权重
    action_token_ids = get_action_token_ids(tokenizer)
    all_ok = True

    for action in ACTION_LABELS:
        expected_weight = ACTION_TOKEN_WEIGHTS[action]
        token_info = action_token_ids[action]

        print(f"\n{action} (expected weight: {expected_weight}):")

        # Only check the actual action tokens (without_space contains the filtered list)
        for tid in token_info['without_space']:
            if tid < vocab_size:
                actual_weight = weights[tid].item()
                status = "OK" if abs(actual_weight - expected_weight) < 0.01 else "FAIL"
                if status == "FAIL":
                    all_ok = False
                decoded = tokenizer.decode([tid])
                print(f"  Token {tid} ('{decoded}'): {actual_weight:.2f} [{status}]")

    print("\n" + "=" * 60)
    print("3. Test Loss Calculation")
    print("=" * 60)

    from torch.nn import CrossEntropyLoss

    # 创建模拟数据
    batch_size = 4
    seq_len = 10

    logits = torch.randn(batch_size, seq_len, vocab_size)

    # 创建 labels，包含一些 action tokens
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    # 获取 forward 和 wait 的 token ID
    forward_ids = action_token_ids['forward']['with_space']
    wait_ids = action_token_ids['wait']['with_space']

    if forward_ids:
        labels[0, 5] = forward_ids[0]
    if wait_ids:
        labels[1, 5] = wait_ids[0]

    # 标准 loss
    loss_fct_standard = CrossEntropyLoss(ignore_index=-100)
    loss_standard = loss_fct_standard(
        logits.view(-1, vocab_size),
        labels.view(-1)
    )

    # 加权 loss
    loss_fct_weighted = CrossEntropyLoss(weight=weights, ignore_index=-100)
    loss_weighted = loss_fct_weighted(
        logits.view(-1, vocab_size),
        labels.view(-1)
    )

    print(f"\nStandard loss: {loss_standard.item():.4f}")
    print(f"Weighted loss: {loss_weighted.item():.4f}")
    print(f"Ratio: {loss_weighted.item() / loss_standard.item():.2f}x")

    print("\n" + "=" * 60)
    print("4. Summary")
    print("=" * 60)

    if all_ok:
        print("\n[SUCCESS] All action token weights are correctly set!")
    else:
        print("\n[WARNING] Some weights may not be correctly set. Please check above.")

    print("\nAction weights summary:")
    for action, weight in ACTION_TOKEN_WEIGHTS.items():
        print(f"  {action}: {weight}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
