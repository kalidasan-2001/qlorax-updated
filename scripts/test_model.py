#!/usr/bin/env python3
"""
Test the fine-tuned QLoRA model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
from pathlib import Path


def load_model():
    """Load the fine-tuned model and tokenizer"""
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "./models/tinyllama-qlora"

    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=False,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=200):
    """Generate a response using the fine-tuned model"""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt) :].strip()


def main():
    """Main testing function"""
    adapter_path = Path("./models/tinyllama-qlora")

    if not adapter_path.exists():
        print("❌ Model not found! Please train the model first.")
        print("Run: python scripts/train_model.py")
        return

    print("🔄 Loading fine-tuned model...")
    try:
        model, tokenizer = load_model()
        print("✅ Model loaded successfully!")

        # Test prompts
        test_prompts = [
            "What is machine learning?",
            "Explain quantum computing simply.",
            "How do you make pasta?",
        ]

        print("\n🧪 Testing the model:")
        print("=" * 50)

        for prompt in test_prompts:
            print(f"\n📝 Prompt: {prompt}")
            response = generate_response(model, tokenizer, prompt)
            print(f"🤖 Response: {response}")
            print("-" * 50)

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
