#!/usr/bin/env python3
"""
Robust QLoRA training script with better error handling and smaller model option
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
    return f"### Input:\n{example['input']}\n\n### Output:\n{example['output']}\n<|endoftext|>"


def download_model_with_retry(model_name, max_retries=3):
    """Download model with retry logic"""
    for attempt in range(max_retries):
        try:
            print(
                f"📥 Attempting to load model (attempt {attempt + 1}/{max_retries})..."
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto",
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            )

            print("✅ Model loaded successfully!")
            return model, tokenizer

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise e
            print("🔄 Retrying in 5 seconds...")
            import time

            time.sleep(5)


def main():
    # Configuration - using a very small model for quick testing
    model_options = [
        "microsoft/DialoGPT-small",  # Much smaller, faster download
        "gpt2",  # Even smaller, very fast
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Original choice
    ]

    print("🚀 Robust QLoRA Training with Transformers + PEFT")
    print("=" * 60)

    # Try models in order of preference
    model, tokenizer = None, None
    for model_name in model_options:
        try:
            print(f"🎯 Trying model: {model_name}")
            model, tokenizer = download_model_with_retry(model_name, max_retries=2)
            print(f"✅ Successfully loaded: {model_name}")
            break
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            continue

    if model is None:
        print("❌ Failed to load any model. Check your internet connection.")
        return 1

    dataset_path = "data/my_custom_dataset.jsonl"
    output_dir = "models/robust-qlora-model"

    # Configure LoRA
    print("⚙️ Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,  # Smaller rank for faster training
        lora_alpha=16,
        lora_dropout=0.05,
        # Use generic target modules that work with most models
        target_modules=(
            ["c_attn", "c_proj", "c_fc"]
            if "gpt2" in str(model.config._name_or_path)
            else ["q_proj", "v_proj"]
        ),
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
            examples["text"],
            truncation=True,
            padding=True,
            max_length=256,  # Shorter for faster training
            return_tensors="pt",
        )

    train_dataset = Dataset.from_dict({"text": train_texts})
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    # Training arguments - optimized for quick training
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,  # Just 1 epoch for quick test
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        learning_rate=5e-4,  # Higher LR for faster convergence
        logging_steps=1,
        save_steps=50,
        evaluation_strategy="no",
        save_strategy="steps",
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None,  # Disable W&B
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
    print(f"Model: {model.config._name_or_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    try:
        trainer.train()
        print("✅ Training completed!")

        # Save the model
        print("💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Model saved to: {output_dir}")

        # Test the model quickly
        print("\n🧪 Quick test of the trained model:")
        test_input = "What is machine learning?"
        inputs = tokenizer.encode(
            f"### Input:\n{test_input}\n\n### Output:\n", return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 Input: {test_input}")
        print(f"🤖 Output: {response}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    print(f"\n🎯 Training completed with exit code: {exit_code}")
    input("Press Enter to exit...")
    exit(exit_code)
