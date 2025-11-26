#!/usr/bin/env python3
"""
Fixed minimal training test with proper data handling
"""

import json
import os

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


def minimal_training_test():
    """Run a minimal training test with proper data handling"""
    print("🧪 Running fixed minimal training test...")

    try:
        # Use a very small model for testing
        model_name = "gpt2"
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
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["c_attn"],
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Create minimal dataset with proper tokenization
        texts = [
            "Hello world, this is a test.",
            "Machine learning is fascinating.",
            "Training language models requires data.",
        ]

        # Tokenize properly
        def tokenize_function(text):
            # Add EOS token to each text
            text_with_eos = text + tokenizer.eos_token
            # Tokenize
            tokenized = tokenizer(
                text_with_eos,
                truncation=True,
                max_length=32,  # Very short for testing
                padding=False,  # Let data collator handle padding
                return_tensors=None,  # Return lists, not tensors yet
            )
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        # Process each text individually
        tokenized_data = []
        for text in texts:
            tokenized_data.append(tokenize_function(text))

        # Create dataset
        dataset = Dataset.from_list(tokenized_data)
        print(f"📊 Dataset created with {len(dataset)} examples")

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        # Minimal training arguments - PROPERLY disable wandb
        training_args = TrainingArguments(
            output_dir="./test_model",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=1000,
            logging_steps=1,
            remove_unused_columns=False,
            report_to=[],  # Disable all reporting including wandb
            logging_dir=None,  # Don't create logs
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        print("🏃 Starting minimal training...")
        train_result = trainer.train()

        print(f"✅ Training completed!")
        print(f"📊 Final loss: {train_result.training_loss:.4f}")

        # Test generation
        print("🧪 Testing generation...")
        model.eval()
        prompt = "Hello"
        inputs = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
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
        print("💡 Key findings:")
        print("   - Model loading: ✅ Working")
        print("   - LoRA setup: ✅ Working")
        print("   - Data tokenization: ✅ Working")
        print("   - Training loop: ✅ Working")
        print("   - Generation: ✅ Working")
        print("\n🚀 You can now run production training with confidence!")
    else:
        print("\n❌ There are still issues with the training pipeline.")
        print("💡 Check the error messages above for details.")
