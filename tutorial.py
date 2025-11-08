#!/usr/bin/env python3
"""
Interactive Fine-Tuning Demo and Tutorial
Shows you exactly how QLoRA fine-tuning works
"""

import json
from pathlib import Path


def show_dataset_format():
    """Show how to format your dataset"""
    print("📊 STEP 1: Dataset Format")
    print("=" * 50)
    print("Your dataset should be in JSONL format (one JSON per line):")
    print()

    examples = [
        {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris.",
        },
        {
            "input": "How do you make coffee?",
            "output": "To make coffee: 1) Boil water, 2) Add coffee grounds, 3) Pour water over grounds, 4) Let steep, 5) Enjoy!",
        },
        {
            "input": "Explain quantum physics simply",
            "output": "Quantum physics studies very tiny particles that behave differently than large objects. They can be in multiple states at once!",
        },
    ]

    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {json.dumps(example, indent=2)}")
        print()


def show_fine_tuning_process():
    """Explain the fine-tuning process"""
    print("🔧 STEP 2: Fine-Tuning Process")
    print("=" * 50)
    print("QLoRA (Quantized Low-Rank Adaptation) works by:")
    print()
    print("1. 📥 Loading a pre-trained model (like TinyLlama)")
    print("2. 🔒 Freezing most model parameters")
    print("3. ➕ Adding small 'adapter' layers (LoRA)")
    print("4. 🎯 Training only the adapter layers on your data")
    print("5. 💾 Saving the adapters (much smaller than full model)")
    print()
    print("Benefits:")
    print("✅ Fast training (only ~1% of parameters)")
    print("✅ Low memory usage (4-bit quantization)")
    print("✅ Easy to share (adapters are small files)")
    print("✅ Can switch between different fine-tunes")
    print()


def show_parameters():
    """Explain key parameters"""
    print("⚙️ STEP 3: Key Parameters")
    print("=" * 50)

    params = {
        "LoRA Rank (r)": {
            "what": "Number of dimensions in adapter layers",
            "values": "16-128 (higher = more capacity, slower)",
            "recommend": "32-64 for most tasks",
        },
        "LoRA Alpha": {
            "what": "Scaling factor for adapter influence",
            "values": "8-32 (higher = stronger adaptation)",
            "recommend": "16 for balanced results",
        },
        "Learning Rate": {
            "what": "How fast the model learns",
            "values": "1e-5 to 1e-3",
            "recommend": "2e-4 for stable training",
        },
        "Epochs": {
            "what": "How many times to see the data",
            "values": "1-10",
            "recommend": "3-5 for most datasets",
        },
        "Batch Size": {
            "what": "Examples processed together",
            "values": "1-16",
            "recommend": "1-4 for limited memory",
        },
    }

    for param, info in params.items():
        print(f"📋 {param}:")
        print(f"   What: {info['what']}")
        print(f"   Range: {info['values']}")
        print(f"   Recommend: {info['recommend']}")
        print()


def show_training_commands():
    """Show available training methods"""
    print("🚀 STEP 4: How to Run Training")
    print("=" * 50)
    print("You have several options:")
    print()

    methods = [
        {
            "name": "Robust Training (Recommended)",
            "command": "./venv/Scripts/python.exe robust_train.py",
            "description": "Handles network issues, tries multiple models",
        },
        {
            "name": "Simple Training",
            "command": "./venv/Scripts/python.exe simple_train.py",
            "description": "Basic training with Transformers + PEFT",
        },
        {
            "name": "Batch File (Windows)",
            "command": "train.bat",
            "description": "Double-click to run, uses Axolotl",
        },
        {
            "name": "Axolotl CLI",
            "command": "./venv/Scripts/python.exe launch_training.py",
            "description": "Full Axolotl features and configuration",
        },
    ]

    for i, method in enumerate(methods, 1):
        print(f"{i}. {method['name']}")
        print(f"   Command: {method['command']}")
        print(f"   Description: {method['description']}")
        print()


def show_monitoring():
    """Show how to monitor training"""
    print("📈 STEP 5: Monitoring Training")
    print("=" * 50)
    print("During training, watch for:")
    print()
    print("✅ Loss decreasing over time")
    print("✅ Learning rate schedule working")
    print("✅ No memory errors")
    print("✅ Reasonable training speed")
    print()
    print("Example training output:")
    print("Step 1: Loss = 2.345, LR = 0.0001")
    print("Step 10: Loss = 1.876, LR = 0.0002")
    print("Step 20: Loss = 1.432, LR = 0.0002")
    print("...")
    print()


def show_testing():
    """Show how to test the model"""
    print("🧪 STEP 6: Testing Your Model")
    print("=" * 50)
    print("After training, test with:")
    print()
    print("./venv/Scripts/python.exe scripts/test_model.py")
    print()
    print("Or create a simple test:")
    print()
    test_code = """
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and adapter
base_model = AutoModelForCausalLM.from_pretrained("model_name")
model = PeftModel.from_pretrained(base_model, "path/to/adapter")

# Test it
prompt = "What is machine learning?"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=100)
response = tokenizer.decode(outputs[0])
print(response)
"""
    print(test_code)


def show_deployment():
    """Show deployment options"""
    print("🚀 STEP 7: Deploy Your Model")
    print("=" * 50)
    print("Once trained, deploy with:")
    print()

    deploy_options = [
        {
            "name": "FastAPI Server",
            "command": "./venv/Scripts/python.exe scripts/api_server.py",
            "access": "http://localhost:8000/docs",
        },
        {
            "name": "Gradio Interface",
            "command": "./venv/Scripts/python.exe scripts/gradio_app.py",
            "access": "http://localhost:7860",
        },
        {
            "name": "Jupyter Notebook",
            "command": "./venv/Scripts/jupyter.exe lab",
            "access": "Interactive development",
        },
    ]

    for option in deploy_options:
        print(f"🔧 {option['name']}:")
        print(f"   Command: {option['command']}")
        print(f"   Access: {option['access']}")
        print()


def main():
    """Main tutorial function"""
    print("🎓 QLORAX Fine-Tuning Tutorial")
    print("=" * 60)
    print("This guide shows you exactly how to fine-tune a language model")
    print("using QLoRA (Quantized Low-Rank Adaptation)")
    print("=" * 60)
    print()

    show_dataset_format()
    print()
    show_fine_tuning_process()
    print()
    show_parameters()
    print()
    show_training_commands()
    print()
    show_monitoring()
    print()
    show_testing()
    print()
    show_deployment()

    print("🎯 SUMMARY")
    print("=" * 50)
    print("1. Prepare your dataset in JSONL format")
    print("2. Choose training method and run it")
    print("3. Monitor training progress")
    print("4. Test your fine-tuned model")
    print("5. Deploy for others to use")
    print()
    print("✨ You're ready to fine-tune! ✨")


if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
