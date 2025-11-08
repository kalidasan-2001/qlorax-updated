#!/usr/bin/env python3
"""
Example script to validate Axolotl configuration and basic functionality.
This script demonstrates how to load and validate the QLoRA configuration.
"""

import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer
import sys


def load_config(config_path):
    """Load and return the Axolotl configuration."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"✓ Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return None


def validate_model_availability(model_name):
    """Check if the base model is accessible."""
    try:
        print(f"🔍 Checking model availability: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Model {model_name} is accessible")
        print(f"  - Vocab size: {tokenizer.vocab_size}")
        print(f"  - Model max length: {tokenizer.model_max_length}")
        return True
    except Exception as e:
        print(f"⚠️  Model {model_name} check failed: {e}")
        print("  This is normal if you haven't downloaded the model yet.")
        return False


def validate_dataset(dataset_path):
    """Check if the dataset file exists and is readable."""
    try:
        if not Path(dataset_path).exists():
            print(f"⚠️  Dataset file not found: {dataset_path}")
            return False

        with open(dataset_path, "r") as f:
            lines = f.readlines()

        print(f"✓ Dataset file found: {dataset_path}")
        print(f"  - Number of examples: {len(lines)}")

        # Show first example
        if lines:
            import json

            first_example = json.loads(lines[0])
            print(f"  - First example keys: {list(first_example.keys())}")

        return True
    except Exception as e:
        print(f"✗ Dataset validation failed: {e}")
        return False


def check_system_requirements():
    """Check system requirements for training."""
    print("\n🖥️  System Requirements Check:")

    # Check Python version
    python_version = sys.version_info
    print(
        f"  - Python version: {python_version.major}.{python_version.minor}.{python_version.micro}"
    )

    # Check PyTorch and CUDA
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("  - Running on CPU (training will be slower)")

    # Check available memory
    import psutil

    memory = psutil.virtual_memory()
    print(f"  - System RAM: {memory.total / 1024**3:.1f} GB")
    print(f"  - Available RAM: {memory.available / 1024**3:.1f} GB")


def main():
    """Main validation function."""
    print("🔧 QLORAX Configuration Validation")
    print("=" * 50)

    # Load configuration
    config_path = Path("configs/default-qlora-config.yml")
    config = load_config(config_path)

    if not config:
        return 1

    # Display key configuration settings
    print(f"\n📋 Configuration Summary:")
    print(f"  - Base model: {config.get('base_model', 'Not specified')}")
    print(f"  - Output directory: {config.get('output_dir', 'Not specified')}")
    print(f"  - Fine-tuning type: {config.get('adapter', 'Not specified')}")
    print(f"  - LoRA rank: {config.get('lora_r', 'Not specified')}")
    print(f"  - LoRA alpha: {config.get('lora_alpha', 'Not specified')}")
    print(f"  - Learning rate: {config.get('learning_rate', 'Not specified')}")
    print(f"  - Num epochs: {config.get('num_epochs', 'Not specified')}")
    print(f"  - Batch size: {config.get('batch_size', 'Not specified')}")

    # Validate model availability
    base_model = config.get("base_model")
    if base_model:
        validate_model_availability(base_model)

    # Validate dataset
    datasets = config.get("datasets", [])
    if datasets and isinstance(datasets, list) and len(datasets) > 0:
        dataset_path = datasets[0].get("path")
        if dataset_path:
            # Convert to absolute path if relative
            if not Path(dataset_path).is_absolute():
                dataset_path = Path.cwd() / dataset_path.lstrip("./")
            validate_dataset(dataset_path)

    # Check system requirements
    check_system_requirements()

    print(f"\n🎯 Next Steps:")
    print(f"  1. Review and customize the configuration in {config_path}")
    print(f"  2. Prepare your training dataset in JSONL format")
    print(f"  3. Run training with: axolotl train {config_path}")
    print(f"  4. Monitor training with Weights & Biases or TensorBoard")

    print(f"\n✨ Configuration validation completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
