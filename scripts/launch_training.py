#!/usr/bin/env python3
"""
Direct training launcher that bypasses shell issues
"""
import os
import subprocess
import sys
from pathlib import Path


def main():
    # Get the project directory
    project_dir = Path(__file__).parent
    venv_python = project_dir / "venv" / "Scripts" / "python.exe"
    config_file = project_dir / "configs" / "default-qlora-config.yml"

    print("🚀 QLORAX QLoRA Training Launcher")
    print("=" * 50)
    print(f"Project: {project_dir}")
    print(f"Python: {venv_python}")
    print(f"Config: {config_file}")
    print("=" * 50)

    # Check if files exist
    if not venv_python.exists():
        print(f"❌ Python executable not found: {venv_python}")
        return 1

    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return 1

    # Change to project directory
    os.chdir(project_dir)

    # Run the training command directly
    cmd = [str(venv_python), "-m", "axolotl.cli.train", str(config_file)]

    print(f"🏃 Running: {' '.join(cmd)}")
    print("=" * 50)

    try:
        # Run the command and stream output
        result = subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n❌ Error running training: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    sys.exit(exit_code)
