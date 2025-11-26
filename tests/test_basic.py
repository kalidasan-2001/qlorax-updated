#!/usr/bin/env python3
"""
Basic test suite for QLoRA Enhanced pipeline
"""

import unittest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class BasicEnvironmentTest(unittest.TestCase):
    """Test basic environment and imports"""
    
    def test_python_version(self):
        """Test Python version is supported"""
        self.assertGreaterEqual(sys.version_info[:2], (3, 9))
    
    def test_project_structure(self):
        """Test basic project structure exists"""
        required_files = [
            "requirements.txt",
            "README.md", 
            "scripts/",
            "configs/",
            "data/"
        ]
        
        for file_path in required_files:
            full_path = project_root / file_path
            self.assertTrue(
                full_path.exists(), 
                f"Required file/directory missing: {file_path}"
            )
    
    def test_basic_imports(self):
        """Test that basic imports work"""
        try:
            import torch
            import transformers
            import datasets
            import peft
        except ImportError as e:
            self.fail(f"Basic import failed: {e}")
    
    def test_scripts_directory(self):
        """Test scripts directory has required files"""
        scripts_dir = project_root / "scripts"
        self.assertTrue(scripts_dir.exists())
        
        # Check for some key script files
        expected_scripts = [
            "enhanced_training.py",
            "train_model.py", 
            "benchmark.py"
        ]
        
        for script in expected_scripts:
            script_path = scripts_dir / script
            if script_path.exists():
                # If it exists, make sure it's a valid Python file
                with open(script_path, 'r') as f:
                    content = f.read()
                    self.assertIn('def', content, f"{script} should contain function definitions")


class ConfigurationTest(unittest.TestCase):
    """Test configuration files"""
    
    def test_requirements_file(self):
        """Test requirements.txt is valid"""
        req_file = project_root / "requirements.txt"
        self.assertTrue(req_file.exists())
        
        with open(req_file, 'r') as f:
            content = f.read()
            # Check for essential packages
            essential_packages = ['torch', 'transformers', 'datasets', 'peft']
            for pkg in essential_packages:
                self.assertIn(pkg, content.lower(), f"Essential package {pkg} not found in requirements")
    
    def test_config_files(self):
        """Test configuration files exist and are valid"""
        config_dir = project_root / "configs"
        if config_dir.exists():
            config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
            if config_files:
                # If config files exist, test they're valid YAML
                import yaml
                for config_file in config_files[:3]:  # Test first 3 configs
                    with open(config_file, 'r') as f:
                        try:
                            yaml.safe_load(f)
                        except yaml.YAMLError:
                            self.fail(f"Invalid YAML in {config_file}")


if __name__ == '__main__':
    unittest.main(verbosity=2)