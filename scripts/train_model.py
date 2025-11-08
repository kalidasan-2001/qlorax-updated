#!/usr/bin/env python3
"""
QLORAX Training Script
Runs Axolotl training with the specified configuration
"""

import sys
import os
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))

try:
    from axolotl.cli.train import do_cli

    def main():
        # Set the config file path
        config_path = project_dir / "configs" / "default-qlora-config.yml"

        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return 1

        print(f"🚀 Starting QLoRA fine-tuning with config: {config_path}")
        print("=" * 60)

        # Prepare arguments for axolotl
        sys.argv = ["axolotl", "train", str(config_path)]

        # Run the training
        try:
            do_cli()
            print("\n✅ Training completed successfully!")
            return 0
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            import traceback

            traceback.print_exc()
            return 1

    if __name__ == "__main__":
        exit_code = main()
        sys.exit(exit_code)

except ImportError as e:
    print(f"❌ Failed to import axolotl: {e}")
    print("\nPlease make sure axolotl is installed:")
    print("pip install axolotl")
    sys.exit(1)
