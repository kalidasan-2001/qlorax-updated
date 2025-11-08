#!/usr/bin/env python3
"""
🎯 QLORAX Simple Demo
A working demonstration of your fine-tuned model
"""

import os
import json
import torch
import time
import warnings
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)


def print_section(title):
    """Print a section header"""
    print(f"\n📋 {title}")
    print("-" * 40)


def load_model_simple():
    """Load model with simple approach"""
    print_section("Loading Fine-Tuned Model")

    try:
        # Use the checkpoint with adapters
        adapter_path = "models/production-model/checkpoints"
        base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or "[PAD]"
        tokenizer.padding_side = "left"
        print("✅ Tokenizer loaded!")

        print("📥 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=False,
        )
        print("✅ Base model loaded!")

        print("🔧 Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ LoRA adapters loaded!")

        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"📊 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        print(f"💡 Efficiency: {trainable_params/total_params*100:.2f}% trainable")

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


def generate_text(model, tokenizer, prompt, max_length=150, temperature=0.7):
    """Generate text with the model"""
    try:
        # Format prompt properly
        formatted_prompt = f"### Input:\n{prompt}\n\n### Output:\n"

        # Tokenize with attention mask
        inputs = tokenizer(
            formatted_prompt, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        print(f"🔄 Generating response...")
        print(f"💭 Prompt: {prompt}")

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        generation_time = time.time() - start_time

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(formatted_prompt, "").strip()

        print(f"🤖 Response: {response}")
        print(f"⏱️  Generated in {generation_time:.2f} seconds")

        return response, generation_time

    except Exception as e:
        print(f"❌ Generation error: {e}")
        return None, 0


def show_model_info():
    """Show model information"""
    print_section("Model Information")

    adapter_config_path = Path(
        "models/production-model/checkpoints/adapter_config.json"
    )
    if adapter_config_path.exists():
        with open(adapter_config_path, "r") as f:
            config = json.load(f)

        print(f"🔧 LoRA Configuration:")
        print(f"   📊 Rank (r): {config.get('r', 'N/A')}")
        print(f"   🎯 Alpha: {config.get('lora_alpha', 'N/A')}")
        print(f"   💧 Dropout: {config.get('lora_dropout', 'N/A')}")
        print(f"   🎯 Target Modules: {len(config.get('target_modules', []))} modules")
        print(f"   📝 Task Type: {config.get('task_type', 'N/A')}")

    # Show file sizes
    adapter_model_path = Path(
        "models/production-model/checkpoints/adapter_model.safetensors"
    )
    if adapter_model_path.exists():
        size_mb = adapter_model_path.stat().st_size / (1024 * 1024)
        print(f"\n📁 Adapter Model Size: {size_mb:.1f} MB")

    print(f"\n🏗️  Architecture: TinyLlama 1.1B + QLoRA")
    print(f"🎯 Fine-tuning: Machine Learning Q&A")
    print(f"💾 Storage: Efficient LoRA adapters")


def demo_capabilities(model, tokenizer):
    """Demo the model's capabilities"""
    print_section("Model Capabilities Demo")

    test_prompts = [
        "What is machine learning?",
        "Explain overfitting in simple terms",
        "How does gradient descent work?",
        "What is the difference between supervised and unsupervised learning?",
        "Write a Python function to calculate mean squared error",
    ]

    total_time = 0
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎯 Test {i}/{len(test_prompts)}")
        print("-" * 30)

        response, gen_time = generate_text(model, tokenizer, prompt)
        total_time += gen_time

        print("-" * 50)

    avg_time = total_time / len(test_prompts)
    print(f"\n📊 Average generation time: {avg_time:.2f} seconds")


def interactive_chat(model, tokenizer):
    """Interactive chat session"""
    print_section("Interactive Chat")
    print("💬 Chat with your fine-tuned model!")
    print("   Type 'quit' to exit")
    print("   Type 'help' for commands")

    while True:
        try:
            user_input = input(f"\n🧑 You: ").strip()

            if user_input.lower() == "quit":
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "help":
                print("📋 Commands:")
                print("   'quit' - Exit chat")
                print("   'help' - Show this help")
                continue
            elif not user_input:
                continue

            print("🤖 Assistant: ", end="", flush=True)
            response, gen_time = generate_text(model, tokenizer, user_input)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main demo function"""
    print_header("QLORAX Simple Demo")
    print("🎯 Your fine-tuned TinyLlama model is ready!")
    print("⚡ Built with QLoRA for efficient fine-tuning")

    # Show model information first
    show_model_info()

    # Load model
    model, tokenizer = load_model_simple()

    if model is None or tokenizer is None:
        print("❌ Failed to load model. Check the logs above.")
        return 1

    # Demo menu
    while True:
        print_header("Demo Options")
        print("1. 🎯 Capability Demo - Test on sample questions")
        print("2. 💬 Interactive Chat - Chat with your model")
        print("3. 🔧 Model Info - Show technical details")
        print("0. 🚪 Exit")

        try:
            choice = input("\n🤔 Choose an option (0-3): ").strip()

            if choice == "1":
                demo_capabilities(model, tokenizer)
            elif choice == "2":
                interactive_chat(model, tokenizer)
            elif choice == "3":
                show_model_info()
            elif choice == "0":
                print("\n👋 Thanks for trying QLORAX!")
                print("🎯 Your model is ready for production!")
                break
            else:
                print("❌ Invalid option. Please choose 0-3.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
