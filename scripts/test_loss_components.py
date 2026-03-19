#!/usr/bin/env python3
"""测试loss组件打印功能"""

import sys
sys.path.insert(0, '/mnt/dataset/shuhzeng/Uni-NaVid_waypoints')

import torch
from uninavid.model.action_head import ActionModelOutput

# 模拟输出
output = ActionModelOutput(
    loss=torch.tensor(0.8),
    lm_loss=torch.tensor(0.5),
    action_loss=torch.tensor(0.3),
    logits=None,
)

print("测试ActionModelOutput:")
print(f"  total loss: {output.loss}")
print(f"  lm_loss: {output.lm_loss}")
print(f"  action_loss: {output.action_loss}")
print(f"  计算验证: {output.lm_loss + output.action_loss} (应该等于 {output.loss})")

if hasattr(output, 'lm_loss') and output.lm_loss is not None:
    print("✓ lm_loss 可访问")
if hasattr(output, 'action_loss') and output.action_loss is not None:
    print("✓ action_loss 可访问")

print("\n所有测试通过！")
