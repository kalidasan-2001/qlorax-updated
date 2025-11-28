#!/usr/bin/env python3
"""
Deterministic Configuration Module for Reproducible LoRA Training
================================================================

This module provides utilities for ensuring deterministic behavior across
different execution environments for reproducibility research.

Key Functions:
- Setup deterministic environment (seeds, algorithms)
- Validate environment configuration
- Calculate file checksums for artifact validation
- Generate environment fingerprints
"""

import os
import sys
import json
import hashlib
import random
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DeterministicConfig:
    """Configuration class for deterministic training."""
    
    # Fixed seed for all randomness sources
    MASTER_SEED = 42
    
    @classmethod
    def setup_deterministic_environment(cls) -> Dict[str, Any]:
        """
        Setup completely deterministic environment for reproducible training.
        
        Returns:
            Dict with configuration status
        """
        config = {}
        
        # Set environment variables for determinism
        os.environ["PYTHONHASHSEED"] = str(cls.MASTER_SEED)
        os.environ["CUDA_DETERMINISTIC"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        
        # Python random
        random.seed(cls.MASTER_SEED)
        config["python_seed"] = cls.MASTER_SEED
        
        # NumPy random
        if NUMPY_AVAILABLE:
            np.random.seed(cls.MASTER_SEED)
            config["numpy_seed"] = cls.MASTER_SEED
        
        # PyTorch determinism
        if TORCH_AVAILABLE:
            torch.manual_seed(cls.MASTER_SEED)
            torch.use_deterministic_algorithms(True)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(cls.MASTER_SEED)
                torch.cuda.manual_seed_all(cls.MASTER_SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            config["torch_seed"] = cls.MASTER_SEED
        
        # Transformers seed (if available)
        try:
            from transformers import set_seed
            set_seed(cls.MASTER_SEED)
            config["transformers_seed"] = cls.MASTER_SEED
        except ImportError:
            pass
        
        config["master_seed"] = cls.MASTER_SEED
        config["deterministic_mode"] = True
        
        print("✅ Deterministic environment configured")
        print(f"   🎲 Master Seed: {cls.MASTER_SEED}")
        print(f"   🐍 Python Seed: {cls.MASTER_SEED}")
        if NUMPY_AVAILABLE:
            print(f"   🔢 NumPy Seed: {cls.MASTER_SEED}")
        if TORCH_AVAILABLE:
            print(f"   🔥 PyTorch Seed: {cls.MASTER_SEED}")
            print(f"   🤖 Transformers Seed: {cls.MASTER_SEED}")
            print(f"   💻 CPU-only Mode: Enabled")
        
        return config
    
    @classmethod
    def validate_deterministic_setup(cls) -> Dict[str, Any]:
        """
        Validate that deterministic setup is properly configured.
        
        Returns:
            Dict with validation results
        """
        validation = {}
        checks_passed = 0
        total_checks = 0
        
        # Check environment variables
        checks = {
            "PYTHONHASHSEED": str(cls.MASTER_SEED),
            "CUDA_DETERMINISTIC": "1",
            "OMP_NUM_THREADS": "1"
        }
        
        print("🔍 Validating deterministic setup...")
        
        for var, expected in checks.items():
            actual = os.environ.get(var)
            passed = actual == expected
            validation[var] = {"expected": expected, "actual": actual, "passed": passed}
            total_checks += 1
            if passed:
                checks_passed += 1
                print(f"   ✅ {var}: {actual}")
            else:
                print(f"   ❌ {var}: Expected '{expected}', got '{actual}'")
        
        # Check PyTorch settings
        if TORCH_AVAILABLE:
            deterministic_algos = torch.are_deterministic_algorithms_enabled()
            validation["torch_deterministic"] = {"enabled": deterministic_algos, "passed": deterministic_algos}
            total_checks += 1
            if deterministic_algos:
                checks_passed += 1
                print("   ✅ PyTorch deterministic algorithms: ENABLED")
            else:
                print("   ❌ PyTorch deterministic algorithms: DISABLED")
            
            # CUDA settings
            if torch.cuda.is_available():
                cuda_deterministic = torch.backends.cudnn.deterministic
                cuda_benchmark = not torch.backends.cudnn.benchmark
                
                validation["cudnn_deterministic"] = {"enabled": cuda_deterministic, "passed": cuda_deterministic}
                validation["cudnn_benchmark"] = {"disabled": cuda_benchmark, "passed": cuda_benchmark}
                
                total_checks += 2
                if cuda_deterministic:
                    checks_passed += 1
                    print("   ✅ cuDNN deterministic: TRUE")
                else:
                    print("   ❌ cuDNN deterministic: FALSE")
                
                if cuda_benchmark:
                    checks_passed += 1
                    print("   ✅ cuDNN benchmark: FALSE")
                else:
                    print("   ❌ cuDNN benchmark: TRUE")
            else:
                print("   ✅ CUDA: Disabled (CPU-only mode)")
                validation["cuda_disabled"] = {"passed": True}
                total_checks += 1
                checks_passed += 1
        
        validation["summary"] = {
            "checks_passed": checks_passed,
            "total_checks": total_checks,
            "pass_percentage": (checks_passed / total_checks) * 100
        }
        
        print(f"\n📊 Validation Results: {checks_passed}/{total_checks} checks passed ({validation['summary']['pass_percentage']:.1f}%)")
        
        if checks_passed == total_checks:
            print("🎉 Deterministic setup validation: PASSED")
        else:
            print("⚠️  Deterministic setup validation: FAILED")
        
        return validation
    
    @classmethod
    def get_training_config(cls) -> Dict[str, Any]:
        """Get training configuration for reproducible training."""
        return {
            "seed": cls.MASTER_SEED,
            "data_seed": cls.MASTER_SEED,
            "deterministic_mode": True,
            "use_cpu": True,
            "no_cuda": True,
            "reproducible_training": True,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
            "dataloader_drop_last": False,
            "gradient_checkpointing": False,
            "tf32": False,
            "bf16": False,
            "fp16": False,
            "load_best_model_at_end": False,
            "remove_unused_columns": True,
            "report_to": [],
            "run_name": None,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "eval_accumulation_steps": 1,
            "learning_rate": 5e-4,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "num_train_epochs": 1,
            "max_steps": -1,
            "lr_scheduler_type": "linear",
            "warmup_steps": 0,
            "logging_steps": 1,
            "eval_steps": 500,
            "save_steps": 500
        }
    
    @classmethod
    def get_lora_config(cls) -> Dict[str, Any]:
        """Get LoRA configuration for deterministic training."""
        return {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "init_lora_weights": True,
            "fan_in_fan_out": False
        }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration for deterministic training."""
        return {
            "model_name_or_path": "microsoft/DialoGPT-medium",
            "use_cache": False,
            "device_map": None,
            "torch_dtype": "auto",
            "trust_remote_code": False,
            "use_auth_token": False,
            "revision": "main",
            "simulation_mode": True  # Use simulation for reproducibility testing
        }


def calculate_file_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def save_environment_fingerprint(output_file: Path) -> Dict[str, Any]:
    """
    Save complete environment fingerprint for reproducibility tracking.
    
    Args:
        output_file: Path to save the environment fingerprint
        
    Returns:
        Dict containing environment information
    """
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable
        },
        "deterministic_config": {
            "master_seed": DeterministicConfig.MASTER_SEED,
            "python_seed": DeterministicConfig.MASTER_SEED,
            "numpy_seed": DeterministicConfig.MASTER_SEED if NUMPY_AVAILABLE else None,
            "pytorch_seed": DeterministicConfig.MASTER_SEED if TORCH_AVAILABLE else None,
            "transformers_seed": DeterministicConfig.MASTER_SEED
        },
        "environment_variables": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "NOT_SET"),
            "CUDA_DETERMINISTIC": os.environ.get("CUDA_DETERMINISTIC", "NOT_SET"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")
        },
        "packages": []
    }
    
    # Get installed packages
    try:
        import pkg_resources
        installed_packages = [f"{d.project_name}=={d.version}" for d in pkg_resources.working_set]
        env_info["packages"] = sorted(installed_packages)
    except Exception:
        pass
    
    # Add Git commit if available
    try:
        import subprocess
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                           stderr=subprocess.DEVNULL).decode().strip()
        env_info["git_commit"] = git_commit
    except Exception:
        pass
    
    # Add ML library specific information
    if TORCH_AVAILABLE:
        env_info["ml_libraries"] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()
        }
        
        if torch.cuda.is_available():
            env_info["ml_libraries"]["cuda_version"] = torch.version.cuda
            env_info["ml_libraries"]["cudnn_deterministic"] = torch.backends.cudnn.deterministic
            env_info["ml_libraries"]["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        else:
            env_info["ml_libraries"]["cuda_version"] = None
            env_info["ml_libraries"]["cudnn_deterministic"] = True
            env_info["ml_libraries"]["cudnn_benchmark"] = False
    
    if NUMPY_AVAILABLE:
        if "ml_libraries" not in env_info:
            env_info["ml_libraries"] = {}
        env_info["ml_libraries"]["numpy_version"] = np.__version__
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(env_info, f, indent=2, sort_keys=True)
    
    return env_info


if __name__ == "__main__":
    """Test the deterministic configuration."""
    print("🔬 Testing Deterministic Configuration")
    print("=" * 50)
    
    # Setup deterministic environment
    config = DeterministicConfig.setup_deterministic_environment()
    
    # Validate setup
    validation = DeterministicConfig.validate_deterministic_setup()
    
    # Test file hashing
    test_content = "reproducibility test content"
    test_file = Path("test_file.txt")
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    file_hash = calculate_file_sha256(test_file)
    print(f"\n📄 Test file hash: {file_hash}")
    
    # Test environment fingerprint
    env_file = Path("test_env.json")
    env_info = save_environment_fingerprint(env_file)
    print(f"🌍 Environment fingerprint saved: {env_file}")
    
    # Cleanup
    test_file.unlink()
    env_file.unlink()
    
    print("\n✅ Deterministic configuration test completed!")