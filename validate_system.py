#!/usr/bin/env python3
"""
QLORAX System Validation & Error Detection Script
Comprehensive check for potential issues
"""

import os
import sys
import json
import yaml
import importlib
from pathlib import Path


def check_environment():
    """Check Python environment and basic setup"""
    print("🔍 Checking Python Environment...")
    print(f"   Python version: {sys.version}")
    print(f"   Python executable: {sys.executable}")

    # Check if we're in virtual environment
    if "venv" in sys.executable:
        print("   ✅ Virtual environment detected")
    else:
        print("   ⚠️  Virtual environment not detected")

    return True


def check_dependencies():
    """Check all required dependencies"""
    print("\n📦 Checking Dependencies...")

    required_packages = [
        "torch",
        "transformers",
        "peft",
        "datasets",
        "yaml",
        "wandb",
        "numpy",
        "matplotlib",
        "seaborn",
        "sklearn",
        "tqdm",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        return False

    print("   ✅ All required packages available")
    return True


def check_files():
    """Check required files and directories"""
    print("\n📁 Checking Files and Directories...")
    
    # Detect if running in GitHub Actions
    is_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'

    required_files = [
        "scripts/train_production.py",
        "scripts/benchmark.py",
        "configs/production-config.yaml",
    ]
    
    # In GitHub Actions, we may have generated data instead of static files
    if not is_github_actions:
        required_files.extend([
            "data/training_data.jsonl",
            "data/test_data.jsonl",
        ])
    else:
        # In GitHub Actions, check for any training data (could be generated)
        print("   ℹ️  GitHub Actions environment detected - checking for generated data")

    required_dirs = ["scripts", "configs", "data", "models"]
    
    # venv is not needed in GitHub Actions (uses system Python)
    if not is_github_actions:
        required_dirs.append("venv")

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")

    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"   ❌ {dir_path}/")
            missing_dirs.append(dir_path)
        else:
            print(f"   ✅ {dir_path}/")

    if missing_files or missing_dirs:
        print(f"\n⚠️  Missing files: {missing_files}")
        print(f"⚠️  Missing directories: {missing_dirs}")
        return False

    return True


def check_data_format():
    """Check data file formats"""
    print("\n📊 Checking Data Format...")
    
    # Check if we're in GitHub Actions
    is_github_actions = os.getenv('GITHUB_ACTIONS', '').lower() == 'true'
    
    if is_github_actions:
        print("   ⚡ GitHub Actions environment detected - checking for generated data...")
        # In GitHub Actions, check for generated synthetic data
        data_files = []
        if os.path.exists("data/"):
            for file in os.listdir("data/"):
                if file.endswith(".jsonl"):
                    data_files.append(f"data/{file}")
        
        if not data_files:
            print("   ❌ No JSONL data files found in GitHub Actions environment")
            return False
            
        print(f"   ✅ Found data files: {data_files}")
        
        # Check first available data file format
        try:
            with open(data_files[0], "r") as f:
                sample_data = [json.loads(line) for line in f]
            
            if sample_data:
                example = sample_data[0]
                print(f"   ✅ Data sample: {len(sample_data)} examples")
                print(f"   ✅ Data keys: {list(example.keys())}")
            else:
                print("   ⚠️ Data file is empty but format is valid")
                
        except Exception as e:
            print(f"   ❌ Error reading data file: {e}")
            return False
            
        return True

    try:
        # Original local environment checks
        # Check training data
        with open("data/training_data.jsonl", "r") as f:
            train_data = [json.loads(line) for line in f]

        print(f"   ✅ Training data: {len(train_data)} examples")

        if train_data:
            example = train_data[0]
            required_keys = ["input", "output"]
            if all(key in example for key in required_keys):
                print(f"   ✅ Data format correct: {list(example.keys())}")
            else:
                print(f"   ❌ Missing required keys: {required_keys}")
                return False

        # Check test data
        with open("data/test_data.jsonl", "r") as f:
            test_data = [json.loads(line) for line in f]

        print(f"   ✅ Test data: {len(test_data)} examples")

        return True

    except Exception as e:
        print(f"   ❌ Data format error: {e}")
        return False


def check_config():
    """Check configuration file"""
    print("\n⚙️ Checking Configuration...")

    try:
        with open("configs/production-config.yaml", "r") as f:
            config = yaml.safe_load(f)

        required_keys = [
            "model_name",
            "data_path",
            "output_dir",
            "lora_r",
            "lora_alpha",
            "learning_rate",
        ]

        missing_keys = []
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
            else:
                print(f"   ✅ {key}: {config[key]}")

        if missing_keys:
            print(f"\n   ❌ Missing config keys: {missing_keys}")
            return False

        print("   ✅ Configuration format correct")
        return True

    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False


def check_model_access():
    """Check if base model can be accessed"""
    print("\n🤖 Checking Model Access...")

    try:
        from transformers import AutoTokenizer

        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        print(f"   🔍 Testing access to {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("   ✅ Model accessible")
        print(f"   📊 Vocab size: {tokenizer.vocab_size}")

        return True

    except Exception as e:
        print(f"   ❌ Model access error: {e}")
        print("   💡 This might be due to network issues or missing authentication")
        return False


def check_gpu():
    """Check GPU availability"""
    print("\n🖥️ Checking GPU Availability...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   📊 GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("   ⚠️  CUDA not available - will use CPU")
            print("   💡 Training will be slower but still functional")

        return True

    except Exception as e:
        print(f"   ❌ GPU check error: {e}")
        return False


def check_disk_space():
    """Check available disk space"""
    print("\n💾 Checking Disk Space...")

    try:
        import shutil

        total, used, free = shutil.disk_usage(".")

        free_gb = free // (1024**3)
        print(f"   📊 Free space: {free_gb} GB")

        if free_gb < 5:
            print("   ⚠️  Low disk space - may cause issues during training")
            return False
        else:
            print("   ✅ Sufficient disk space available")

        return True

    except Exception as e:
        print(f"   ❌ Disk space check error: {e}")
        return False


def run_simple_test():
    """Run a simple functionality test"""
    print("\n🧪 Running Simple Functionality Test...")

    try:
        # Test basic imports
        from transformers import AutoTokenizer
        import torch

        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Smaller model for testing
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)

        print(
            f"   ✅ Tokenization test: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'"
        )

        # Test tensor operations
        x = torch.randn(2, 3)
        y = torch.mm(x, x.T)
        print(f"   ✅ Tensor operations: {x.shape} -> {y.shape}")

        print("   ✅ Basic functionality working")
        return True

    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        return False


def main():
    """Run comprehensive validation"""
    print("🔬 QLORAX System Validation & Error Detection")
    print("=" * 60)

    checks = [
        ("Environment", check_environment),
        ("Dependencies", check_dependencies),
        ("Files", check_files),
        ("Data Format", check_data_format),
        ("Configuration", check_config),
        ("Model Access", check_model_access),
        ("GPU", check_gpu),
        ("Disk Space", check_disk_space),
        ("Functionality", run_simple_test),
    ]

    passed = 0
    failed = 0

    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ {name} check failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("\n🎉 All checks passed! Your system is ready for training.")
        print("\n🚀 Next steps:")
        print("   1. Run: python quick_start.py")
        print(
            "   2. Or: python scripts/train_production.py --config configs/production-config.yaml"
        )
    else:
        print(f"\n⚠️  {failed} issues found. Please address them before training.")
        print("\n💡 Common solutions:")
        print("   - Install missing packages: pip install <package_name>")
        print("   - Check internet connection for model downloads")
        print("   - Ensure sufficient disk space")
        print("   - Verify file paths and permissions")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
