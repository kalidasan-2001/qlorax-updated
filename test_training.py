#!/usr/bin/env python3
"""
Minimal working fine-tuning test to identify potential issues
"""

import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def minimal_training_test():
    """Run a minimal training test to validate the pipeline"""
    print("🧪 Running minimal training test...")

    try:
        # Use a very small model for testing
        model_name = "gpt2"  # Much smaller than TinyLlama
        print(f"📥 Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
        )

        print(f"✅ Model loaded: {model.num_parameters():,} parameters")

        # Setup minimal LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # Very small rank
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["c_attn"],  # Only target one module type
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Create minimal dataset
        data = [
            {"text": "Hello world"},
            {"text": "This is a test"},
            {"text": "Machine learning is fun"},
        ]

        def tokenize_function(examples):
            # Tokenize and set up labels for language modeling
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # Don't pad here, let data collator handle it
                max_length=50,  # Very short
            )
            # For language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = Dataset.from_dict({"text": [item["text"] for item in data]})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Minimal training arguments
        training_args = TrainingArguments(
            output_dir="./test_model",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=1000,  # Don't save during this test
            logging_steps=1,
            remove_unused_columns=False,
            report_to=[],  # Disable wandb for test - use empty list instead of None
        )

        from transformers import DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # This is for causal language modeling, not masked
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        print("🏃 Starting minimal training (1 epoch, 3 examples)...")
        train_result = trainer.train()

        print(f"✅ Training completed!")
        print(f"📊 Final loss: {train_result.training_loss:.4f}")

        # Test generation
        print("🧪 Testing generation...")
        prompt = "Hello"
        inputs = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 Generated: '{generated}'")

        # Clean up
        import shutil

        if os.path.exists("./test_model"):
            shutil.rmtree("./test_model")

        print("🎉 Minimal training test PASSED!")
        return True

    except Exception as e:
        print(f"❌ Minimal training test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = minimal_training_test()
    if success:
        print("\n✅ The training pipeline is working correctly!")
        print("💡 You can now run production training with confidence.")
    else:
        print("\n❌ There are issues with the training pipeline.")
        print("💡 Check the error messages above for details.")
