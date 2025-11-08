#!/usr/bin/env python3
"""
Install additional dependencies for benchmarking and production training
"""

import subprocess
import sys
from pathlib import Path


def install_package(package_name, description=""):
    """Install a package with error handling"""
    try:
        print(f"📦 Installing {package_name}... {description}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"✅ {package_name} installed successfully")
        else:
            print(f"❌ Failed to install {package_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing {package_name}: {e}")
        return False
    return True


def main():
    """Install additional dependencies"""
    print("🚀 Installing additional dependencies for QLORAX...")

    packages = [
        ("nltk", "Natural Language Toolkit for BLEU scores"),
        ("rouge-score", "ROUGE metric evaluation"),
        ("sentence-transformers", "Semantic similarity evaluation"),
        ("matplotlib", "Plotting and visualization"),
        ("seaborn", "Statistical visualization"),
        ("scikit-learn", "Machine learning utilities"),
        ("wandb", "Weights & Biases logging"),
        ("pyyaml", "YAML configuration files"),
        ("tqdm", "Progress bars"),
    ]

    success_count = 0
    for package, description in packages:
        if install_package(package, description):
            success_count += 1

    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{len(packages)} packages")

    if success_count == len(packages):
        print("🎉 All dependencies installed successfully!")
        print("🚀 You're ready to run production training and benchmarking!")
    else:
        print("⚠️  Some packages failed to install. Check the output above.")
        print("💡 You can try installing them manually with pip install <package_name>")

    print("\n📋 Next steps:")
    print(
        "1. Run production training: python scripts/train_production.py --config configs/production-config.yaml"
    )
    print(
        "2. Run benchmarking: python scripts/benchmark.py --model models/your-model --test-data data/test_data.jsonl --output results/"
    )
    print("3. Check the comprehensive guide: COMPREHENSIVE_GUIDE.md")


if __name__ == "__main__":
    main()
