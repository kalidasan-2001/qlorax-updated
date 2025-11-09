#!/usr/bin/env python3
"""
Simple QLoRA training script using Transformers and PEFT directly
This bypasses axolotl CLI issues
"""

import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path):
    """Load JSONL dataset"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_training_prompt(example):
    """Create training prompt from input/output pair"""
    return f"### Input:\n{example['input']}\n\n### Output:\n{example['output']}\n"


def main():
    # Configuration
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_path = "data/my_custom_dataset.jsonl"
    output_dir = "models/tinyllama-qlora-simple"

    print("🚀 Simple QLoRA Training with Transformers + PEFT")
    print("=" * 60)

    # Load tokenizer and model
    print("📥 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=False,
    )

    # Configure LoRA
    print("⚙️ Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    print("📊 Loading dataset...")
    raw_data = load_dataset(dataset_path)

    # Create training prompts
    train_texts = [create_training_prompt(example) for example in raw_data]

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding=True, max_length=512
        )

    train_dataset = Dataset.from_dict({"text": train_texts})
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        save_strategy="steps",
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Start training
    print("🏃 Starting training...")
    print(f"Dataset size: {len(train_texts)} examples")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    try:
        trainer.train()
        print("✅ Training completed!")

        # Save the model
        print("💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    exit(exit_code)
