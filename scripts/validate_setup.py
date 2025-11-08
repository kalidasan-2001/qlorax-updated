#!/usr/bin/env python3
"""
QLORAX Project Setup Validation Script
Tests that all core dependencies are working correctly.
"""

import sys
import importlib
from pathlib import Path


def test_import(module_name, description):
    """Test importing a module and return success status."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {description}")
        return True
    except ImportError as e:
        print(f"✗ {description} - {e}")
        return False
    except Exception as e:
        print(f"⚠ {description} - {e}")
        return False


def main():
    """Run all import tests."""
    print("🚀 QLORAX Project Setup Validation")
    print("=" * 50)

    # Test core packages
    tests = [
        ("torch", "PyTorch - Deep Learning Framework"),
        ("transformers", "Hugging Face Transformers"),
        ("fastapi", "FastAPI - Web Framework"),
        ("gradio", "Gradio - UI Framework"),
        ("axolotl", "Axolotl - Fine-tuning Framework"),
        ("pandas", "Pandas - Data Analysis"),
        ("numpy", "NumPy - Scientific Computing"),
        ("matplotlib", "Matplotlib - Plotting"),
        ("seaborn", "Seaborn - Statistical Visualization"),
        ("jupyter", "Jupyter - Interactive Computing"),
        ("einops", "Einops - Tensor Operations"),
        ("fire", "Fire - CLI Framework"),
        ("sentencepiece", "SentencePiece - Tokenization"),
        ("wandb", "Weights & Biases - Experiment Tracking"),
        ("tensorboard", "TensorBoard - Visualization"),
        ("black", "Black - Code Formatter"),
        ("pytest", "Pytest - Testing Framework"),
    ]

    success_count = 0
    total_tests = len(tests)

    for module, description in tests:
        if test_import(module, description):
            success_count += 1

    print("\n" + "=" * 50)
    print(f"📊 Results: {success_count}/{total_tests} packages imported successfully")

    # Check project structure
    print("\n🏗️  Project Structure Check:")
    project_root = Path.cwd()
    expected_dirs = ["configs", "data", "notebooks", "scripts"]
    expected_files = [".gitignore", "README.md", "requirements.txt"]

    for dir_name in expected_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ Directory: {dir_name}/")
        else:
            print(f"✗ Directory: {dir_name}/ - Missing")

    for file_name in expected_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✓ File: {file_name}")
        else:
            print(f"✗ File: {file_name} - Missing")

    # Display Python environment info
    print(f"\n🐍 Python Version: {sys.version}")
    print(f"📍 Current Directory: {project_root}")

    if success_count >= total_tests * 0.8:  # 80% success rate
        print("\n🎉 Project setup looks good! You can start developing.")
        return 0
    else:
        print(
            f"\n⚠️  Some packages are missing. Run 'pip install -r requirements.txt' to install missing dependencies."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
