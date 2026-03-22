#!/usr/bin/env python3
"""
评估模型在验证集上的预测动作分布（简化版）
直接使用OmniNavActionDataset加载验证集
"""

import os
import sys
import json
import argparse
from collections import Counter
from typing import List

import torch
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uninavid.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.train.omninav_action_dataset import OmniNavActionDataset, OmniNavActionDataArguments


def parse_actions_from_output(output: str) -> List[str]:
    """从模型输出中解析动作序列"""
    valid_actions = {'forward', 'left', 'right', 'wait', 'stop'}
    actions = []

    for word in output.lower().split():
        word = word.strip('.,!?;:')
        if word in valid_actions:
            actions.append('wait' if word == 'stop' else word)

    return actions


class ActionPredictor:
    """动作预测器"""

    def __init__(self, base_model_path: str, lora_path: str):
        print(f"加载基础模型: {base_model_path}")
        print(f"加载LoRA权重: {lora_path}")

        model_name = get_model_name_from_path(base_model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            lora_path, base_model_path, model_name
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

        print("模型加载完成\n")

    def reset(self):
        """重置模型状态"""
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

    def predict(self, video_path: str, instruction: str) -> str:
        """预测动作"""
        self.reset()

        # 加载视频帧
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return ""

        # 处理视频
        batch_image = np.asarray(frames)
        self.model.get_model().new_frames = len(frames)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values']
        video = video.half().cuda()

        # 构建prompt
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

        with torch.inference_mode():
            self.model.update_prompt([[question]])
            output_ids = self.model.generate(
                input_ids,
                images=[video],
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


def evaluate_on_dataset(
    base_model_path: str,
    lora_path: str,
    data_root: str,
    video_root: str,
    max_samples: int = 50
):
    """在验证集上评估模型预测分布"""

    print("="*60)
    print("评估模型在训练集上的预测动作分布")
    print("="*60)
    print()

    # 1. 加载模型
    predictor = ActionPredictor(base_model_path, lora_path)

    # 2. 加载训练集
    print("加载训练集...")
    data_args = OmniNavActionDataArguments(
        action_root=data_root,
        video_root=video_root,
        split='train',
        enable_oversampling=False,
        image_processor=predictor.image_processor,
    )

    dataset = OmniNavActionDataset(
        tokenizer=predictor.tokenizer,
        data_args=data_args
    )

    print(f"训练集共有 {len(dataset)} 个样本")
    print(f"将评估前 {max_samples} 个样本\n")

    # 3. 统计ground truth和预测的动作分布
    gt_action_counter = Counter()
    pred_action_counter = Counter()

    eval_samples = min(max_samples, len(dataset))

    for i in tqdm(range(eval_samples), desc="评估样本"):
        try:
            # 获取样本的原始数据
            ep_key, sample_idx, sample_data = dataset.samples[i]

            # 从sample_data中提取信息
            instruction = sample_data.get('instruction', '')
            gt_actions = sample_data.get('actions', [])

            # 构建视频路径
            # ep_key格式: "inst_type/agent_type/scene/episode"
            parts = ep_key.split('/')
            if len(parts) == 4:
                inst_type, agent_type, scene, episode = parts
                # 视频路径不包含inst_type
                video_path = os.path.join(video_root, 'train', agent_type, scene, episode, 'rgb.mp4')
            else:
                print(f"\n警告：无效的episode key格式: {ep_key}")
                continue

            # 统计ground truth
            for action in gt_actions:
                gt_action_counter[action] += 1

            # 运行推理
            if not os.path.exists(video_path):
                print(f"\n警告：视频文件不存在: {video_path}")
                continue

            output = predictor.predict(video_path, instruction)
            pred_actions = parse_actions_from_output(output)

            # Print first 5 samples for debugging
            if i < 5:
                print(f"\n样本 {i}:")
                print(f"  指令: {instruction[:100]}...")
                print(f"  模型输出: {output}")
                print(f"  解析的动作: {pred_actions}")
                print(f"  Ground Truth: {gt_actions}")

            for action in pred_actions:
                pred_action_counter[action] += 1

        except Exception as e:
            print(f"\n警告：样本 {i} 评估失败: {e}")
            continue

    # 4. 输出结果
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}\n")

    print(f"成功评估的样本: {eval_samples}\n")

    gt_total = sum(gt_action_counter.values())
    pred_total = sum(pred_action_counter.values())

    print("Ground Truth动作分布:")
    for action in ['forward', 'left', 'right', 'wait', 'stop']:
        count = gt_action_counter.get(action, 0)
        pct = count / gt_total * 100 if gt_total > 0 else 0
        print(f"  {action:8s}: {count:6d} ({pct:5.2f}%)")

    print(f"\n模型预测动作分布:")
    for action in ['forward', 'left', 'right', 'wait', 'stop']:
        count = pred_action_counter.get(action, 0)
        pct = count / pred_total * 100 if pred_total > 0 else 0
        print(f"  {action:8s}: {count:6d} ({pct:5.2f}%)")

    print(f"\n{'='*60}")
    print("分析")
    print(f"{'='*60}\n")

    gt_forward_pct = gt_action_counter.get('forward', 0) / gt_total * 100 if gt_total > 0 else 0
    pred_forward_pct = pred_action_counter.get('forward', 0) / pred_total * 100 if pred_total > 0 else 0

    print(f"Ground Truth中forward占比: {gt_forward_pct:.1f}%")
    print(f"模型预测中forward占比: {pred_forward_pct:.1f}%\n")

    if pred_forward_pct > 80:
        print("❌ 模型预测仍然严重偏向forward (>80%)")
        print("   Oversample策略可能不够有效，需要进一步调整")
    elif pred_forward_pct > 70:
        print("⚠️  模型预测偏向forward (70-80%)")
        print("   有一定改善，但可能需要进一步调整oversample参数")
    elif pred_forward_pct > 60:
        print("✓ 模型预测的forward占比合理 (60-70%)")
        print("  Oversample策略有效，模型学到了更平衡的动作分布")
    else:
        print("✓✓ 模型预测的动作分布很平衡 (<60% forward)")
        print("   Oversample策略非常有效！")

    # 保存结果
    result = {
        'samples_evaluated': eval_samples,
        'ground_truth_distribution': dict(gt_action_counter),
        'predicted_distribution': dict(pred_action_counter),
        'gt_forward_pct': gt_forward_pct,
        'pred_forward_pct': pred_forward_pct,
    }

    output_file = 'model_prediction_distribution.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='评估模型预测动作分布')
    parser.add_argument(
        '--base_model',
        type=str,
        default='/mnt/dataset/shuhzeng/model/uninavid-7b-full-224-video-fps-1-grid-2',
        help='基础模型路径'
    )
    parser.add_argument(
        '--lora_path',
        type=str,
        default='/mnt/dataset/shuhzeng/Uni-NaVid_waypoints/checkpoints/omninav_action_lora_official_319/checkpoint-7500',
        help='LoRA checkpoint路径'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/mnt/dataset/shuhzeng/OmniNavBench/OmniNavBenchActionData',
        help='数据根目录'
    )
    parser.add_argument(
        '--video_root',
        type=str,
        default='/mnt/dataset/shuhzeng/OmniNavBench/OmniNavBenchVideos',
        help='视频根目录'
    )
    parser.add_argument(
        '--max_samples',
        type=str,
        default=50,
        help='最多评估多少个样本'
    )

    args = parser.parse_args()

    evaluate_on_dataset(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        data_root=args.data_root,
        video_root=args.video_root,
        max_samples=int(args.max_samples)
    )


if __name__ == '__main__':
    main()
