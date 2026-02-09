# -*- coding: utf-8 -*-
"""
QLoRA 微调 Deepseek-7B

在 Google Colab T4 GPU 上微调 Deepseek-7B 模型

使用方法（Colab）：
    !python scripts/qlora/train_qlora.py \
        --model_name deepseek-ai/deepseek-llm-7b-chat \
        --data_path data/qlora/instructions.jsonl \
        --output_dir /content/drive/MyDrive/Graduation_Project/qlora_output \
        --num_epochs 3 \
        --batch_size 4 \
        --learning_rate 2e-4
"""
import os
import json
import torch
from pathlib import Path
import argparse
from datetime import datetime

# 导入 transformers 和 PEFT
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import bitsandbytes as bnb


def load_instruction_dataset(data_path: str, tokenizer, max_length: int = 512):
    """
    加载指令数据集并进行 tokenize
    
    Args:
        data_path: 数据文件路径
        tokenizer: Tokenizer
        max_length: 最大长度
    
    Returns:
        处理后的数据集
    """
    # 加载 JSONL 数据
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    def format_instruction(example):
        """格式化指令为对话格式"""
        instruction = example['instruction']
        input_text = example['input']
        output_text = example['output']
        
        # Deepseek 对话格式
        prompt = f"User: {instruction}\n{input_text}\n\nAssistant: {output_text}"
        
        return {'text': prompt}
    
    # 格式化数据
    dataset = dataset.map(format_instruction)
    
    def tokenize_function(examples):
        """Tokenize 函数"""
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def print_trainable_parameters(model):
    """
    打印可训练参数数量
    
    Args:
        model: 模型
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"可训练参数: {trainable_params:,} || "
        f"总参数: {all_param:,} || "
        f"可训练比例: {100 * trainable_params / all_param:.2f}%"
    )


def main():
    parser = argparse.ArgumentParser(description='QLoRA 微调 Deepseek-7B')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, 
                        default='deepseek-ai/deepseek-llm-7b-chat',
                        help='基础模型名称')
    parser.add_argument('--data_path', type=str, 
                        default='data/qlora/instructions.jsonl',
                        help='训练数据路径')
    parser.add_argument('--output_dir', type=str, 
                        default='qlora_output',
                        help='输出目录')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批大小')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    
    # LoRA 参数
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    
    args = parser.parse_args()
    
    print("="*60)
    print("QLoRA 微调 Deepseek-7B")
    print("="*60)
    print(f"基础模型: {args.model_name}")
    print(f"训练数据: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print("="*60)
    
    # 1. 加载 Tokenizer
    print("\n1. 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("   ✓ Tokenizer 加载成功")
    
    # 2. 加载模型（4-bit 量化）
    print("\n2. 加载模型（4-bit 量化）...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    print("   ✓ 模型加载成功")
    
    # 3. 准备模型用于训练
    print("\n3. 准备模型用于 k-bit 训练...")
    model = prepare_model_for_kbit_training(model)
    print("   ✓ 模型准备完成")
    
    # 4. 配置 LoRA
    print("\n4. 配置 LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],  # Deepseek 的注意力层
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    print("   ✓ LoRA 配置完成")
    
    # 5. 加载数据集
    print("\n5. 加载并处理数据集...")
    train_dataset = load_instruction_dataset(
        args.data_path,
        tokenizer,
        args.max_length
    )
    print(f"   ✓ 数据集加载成功，共 {len(train_dataset)} 条样本")
    
    # 6. 配置训练参数
    print("\n6. 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        warmup_steps=50,
        lr_scheduler_type='cosine',
        optim='paged_adamw_8bit',
        report_to='none'
    )
    print("   ✓ 训练参数配置完成")
    
    # 7. 创建 Trainer
    print("\n7. 创建 Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    print("   ✓ Trainer 创建成功")
    
    # 8. 开始训练
    print("\n8. 开始训练...")
    print("="*60)
    trainer.train()
    print("="*60)
    print("   ✓ 训练完成")
    
    # 9. 保存模型
    print("\n9. 保存模型...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"   ✓ 模型已保存到 {args.output_dir}")
    
    # 10. 保存训练信息
    print("\n10. 保存训练信息...")
    training_info = {
        'model_name': args.model_name,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'num_samples': len(train_dataset),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    info_path = Path(args.output_dir) / 'training_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    print(f"   ✓ 训练信息已保存到 {info_path}")
    
    print("\n" + "="*60)
    print("QLoRA 微调完成！")
    print("="*60)
    print(f"输出目录: {args.output_dir}")
    print(f"LoRA 权重文件: adapter_model.bin")
    print(f"配置文件: adapter_config.json")
    print("="*60)


if __name__ == '__main__':
    main()
