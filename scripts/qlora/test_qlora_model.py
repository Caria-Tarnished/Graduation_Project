# -*- coding: utf-8 -*-
"""
测试 QLoRA 微调后的模型

本地测试脚本（需要下载 LoRA 权重文件）

使用方法：
    python scripts/qlora/test_qlora_model.py \
        --adapter_path models/qlora/adapter \
        --test_cases "美联储宣布加息25个基点"
"""
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model_name: str, adapter_path: str):
    """
    加载微调后的模型
    
    Args:
        base_model_name: 基础模型名称
        adapter_path: LoRA 权重路径
    
    Returns:
        model, tokenizer
    """
    print(f"加载基础模型: {base_model_name}...")
    
    # 加载基础模型（4-bit 量化）
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    
    print(f"加载 LoRA 权重: {adapter_path}...")
    
    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    
    print("✓ 模型加载完成")
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7
):
    """
    生成回复
    
    Args:
        model: 模型
        tokenizer: Tokenizer
        instruction: 指令
        input_text: 输入文本
        max_new_tokens: 最大生成长度
        temperature: 温度参数
    
    Returns:
        生成的回复
    """
    # 构建 prompt
    prompt = f"User: {instruction}\n{input_text}\n\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 Assistant 的回复
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(description='测试 QLoRA 微调后的模型')
    
    parser.add_argument('--base_model', type=str,
                        default='deepseek-ai/deepseek-llm-7b-chat',
                        help='基础模型名称')
    parser.add_argument('--adapter_path', type=str,
                        default='models/qlora/adapter',
                        help='LoRA 权重路径')
    parser.add_argument('--test_cases', type=str, nargs='+',
                        help='测试案例（可以提供多个）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("QLoRA 微调模型测试")
    print("="*60)
    
    # 加载模型
    model, tokenizer = load_model(args.base_model, args.adapter_path)
    
    # 默认测试案例
    if not args.test_cases:
        test_cases = [
            {
                "instruction": "分析以下财经快讯对市场的影响",
                "input": "美联储宣布加息25个基点"
            },
            {
                "instruction": "解释什么是预期兑现",
                "input": "市场前期已经大涨，利好消息发布后反而下跌，这是为什么？"
            },
            {
                "instruction": "根据财报数据回答问题",
                "input": "黄金市场2023年的表现如何？"
            }
        ]
    else:
        # 使用用户提供的测试案例
        test_cases = [
            {
                "instruction": "分析以下财经快讯对市场的影响",
                "input": case
            }
            for case in args.test_cases
        ]
    
    # 测试
    print("\n开始测试...")
    print("="*60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- 测试案例 {i} ---")
        print(f"Instruction: {test['instruction']}")
        print(f"Input: {test['input']}")
        print(f"\n生成中...")
        
        response = generate_response(
            model,
            tokenizer,
            test['instruction'],
            test['input']
        )
        
        print(f"\nOutput:\n{response}")
        print("-"*60)
    
    print("\n✓ 测试完成")


if __name__ == '__main__':
    main()
