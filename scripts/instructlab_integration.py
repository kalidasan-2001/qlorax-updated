#!/usr/bin/env python3
"""
QLORAX InstructLab Integration
Adds synthetic data generation and knowledge injection capabilities to QLORAX
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import subprocess
import sys

# Try to import InstructLab components
try:
    # Note: These imports will be available after InstructLab installation
    # from instructlab.config import Config
    # from instructlab.data.generate import generate_data
    # from instructlab.model.serve import serve_model
    # from instructlab.training import train_model
    INSTRUCTLAB_AVAILABLE = False  # Set to True when installed
    logging.warning(
        "InstructLab not yet installed. Run: pip install -r requirements-instructlab.txt"
    )
except ImportError:
    INSTRUCTLAB_AVAILABLE = False
    logging.warning("InstructLab not installed. Install with: pip install instructlab")

logger = logging.getLogger(__name__)


class QLORAXInstructLab:
    """InstructLab integration for QLORAX synthetic data generation and knowledge injection"""

    def __init__(self, config_path: str = "configs/instructlab-config.yaml"):
        """Initialize InstructLab integration

        Args:
            config_path: Path to InstructLab configuration file
        """
        self.config_path = Path(config_path)
        self.project_root = Path(__file__).parent.parent
        self.taxonomy_path = self.project_root / "instructlab" / "taxonomy"
        self.generated_data_path = self.project_root / "data" / "instructlab_generated"
        self.knowledge_path = self.project_root / "instructlab" / "knowledge"

        # Load configuration
        self.config = self.load_config()

        # Setup directories
        self.setup_directories()

        logger.info(f"QLORAX InstructLab integration initialized")
        logger.info(f"Taxonomy path: {self.taxonomy_path}")
        logger.info(f"Generated data path: {self.generated_data_path}")

    def load_config(self) -> Dict[str, Any]:
        """Load InstructLab configuration"""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return {
                "general": {"log_level": "INFO"},
                "data_generation": {
                    "model_name": "microsoft/DialoGPT-medium",
                    "num_samples": 100,
                    "batch_size": 10,
                    "max_length": 512,
                },
                "taxonomy": {
                    "base_path": "instructlab/taxonomy",
                    "domains": ["general", "technical", "conversational"],
                },
                "training": {
                    "merge_with_existing": True,
                    "existing_data_weight": 0.7,
                    "synthetic_data_weight": 0.3,
                },
                "output": {
                    "data_dir": "data/instructlab_generated",
                    "combined_data_file": "data/qlorax_instructlab_combined.jsonl",
                },
            }

    def setup_directories(self):
        """Create necessary directories for InstructLab"""
        directories = [
            self.taxonomy_path,
            self.generated_data_path,
            self.knowledge_path,
            self.project_root / "instructlab" / "models",
            self.project_root / "instructlab" / "config",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info("InstructLab directories created")

    def check_instructlab_installation(self) -> bool:
        """Check if InstructLab is properly installed"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import instructlab"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def install_instructlab(self) -> bool:
        """Install InstructLab and dependencies"""
        logger.info("Installing InstructLab...")

        requirements_file = self.project_root / "requirements-instructlab.txt"
        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return False

        try:
            # Install InstructLab requirements
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True,
                capture_output=True,
                text=True,
            )

            logger.info("InstructLab installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install InstructLab: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False

    def create_taxonomy_structure(self, domain: str = "qlorax_custom") -> Path:
        """Create initial taxonomy structure for data generation

        Args:
            domain: Domain name for the taxonomy

        Returns:
            Path to created taxonomy file
        """
        taxonomy_data = {
            "version": 1,
            "domain": domain,
            "created_by": "QLORAX",
            "description": f"Custom taxonomy for {domain} domain knowledge",
            "seed_examples": [
                {
                    "question": "What is machine learning?",
                    "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
                },
                {
                    "question": "Explain QLoRA fine-tuning",
                    "answer": "QLoRA (Quantized Low-Rank Adaptation) is an efficient fine-tuning method that uses 4-bit quantization and low-rank adapters to reduce memory usage while maintaining performance.",
                },
                {
                    "question": "How does parameter-efficient fine-tuning work?",
                    "answer": "Parameter-efficient fine-tuning methods like LoRA only update a small subset of model parameters, making fine-tuning faster and more memory-efficient while preserving model capabilities.",
                },
            ],
            "knowledge_areas": [
                "machine_learning_fundamentals",
                "fine_tuning_techniques",
                "model_optimization",
                "deep_learning_concepts",
            ],
        }

        taxonomy_file = self.taxonomy_path / f"{domain}.yaml"
        with open(taxonomy_file, "w") as f:
            yaml.dump(taxonomy_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created taxonomy structure at {taxonomy_file}")
        return taxonomy_file

    def generate_synthetic_data(
        self,
        taxonomy_path: Optional[str] = None,
        num_samples: int = None,
        model_name: str = None,
    ) -> Path:
        """Generate synthetic training data using InstructLab methodology

        Args:
            taxonomy_path: Path to taxonomy file
            num_samples: Number of samples to generate
            model_name: Model to use for generation

        Returns:
            Path to generated data file
        """
        if not self.check_instructlab_installation():
            logger.warning("InstructLab not installed. Using mock data generation.")
            return self._generate_mock_data(num_samples or 50)

        # Use config defaults if not specified
        num_samples = num_samples or self.config["data_generation"]["num_samples"]
        model_name = model_name or self.config["data_generation"]["model_name"]
        taxonomy_path = taxonomy_path or (self.taxonomy_path / "qlorax_custom.yaml")

        # Ensure taxonomy exists
        if not Path(taxonomy_path).exists():
            taxonomy_path = self.create_taxonomy_structure()

        output_file = (
            self.generated_data_path
            / f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

        try:
            logger.info(
                f"Generating {num_samples} synthetic samples using {model_name}..."
            )

            # TODO: Implement actual InstructLab data generation when available
            # This is a placeholder for the actual implementation
            generated_samples = self._simulate_instructlab_generation(
                taxonomy_path, num_samples, model_name
            )

            # Convert to QLORAX format (input/output pairs)
            qlorax_samples = []
            for sample in generated_samples:
                qlorax_samples.append(
                    {
                        "input": sample.get("question", sample.get("instruction", "")),
                        "output": sample.get("answer", sample.get("response", "")),
                        "source": "instructlab_synthetic",
                        "domain": sample.get("domain", "general"),
                        "generated_at": datetime.now().isoformat(),
                    }
                )

            # Save in JSONL format compatible with QLORAX
            with open(output_file, "w", encoding="utf-8") as f:
                for sample in qlorax_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            logger.info(
                f"Generated {len(qlorax_samples)} samples saved to {output_file}"
            )
            return output_file

        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            # Fallback to mock data
            return self._generate_mock_data(num_samples)

    def _generate_mock_data(self, num_samples: int) -> Path:
        """Generate mock synthetic data for testing purposes"""
        logger.info(f"Generating {num_samples} mock synthetic samples...")

        mock_templates = [
            {
                "input": "What is the purpose of {concept} in machine learning?",
                "output": "{concept} is an important technique in machine learning that helps improve model performance and efficiency.",
                "concepts": [
                    "regularization",
                    "normalization",
                    "feature engineering",
                    "cross-validation",
                    "ensemble methods",
                ],
            },
            {
                "input": "How does {technique} work in deep learning?",
                "output": "{technique} is a method used in deep learning to enhance model training and performance through specific algorithmic approaches.",
                "concepts": [
                    "dropout",
                    "batch normalization",
                    "attention mechanism",
                    "residual connections",
                    "transfer learning",
                ],
            },
            {
                "input": "Explain the benefits of {method} for model optimization",
                "output": "{method} provides significant advantages for model optimization by improving efficiency and reducing computational requirements.",
                "concepts": [
                    "gradient checkpointing",
                    "mixed precision training",
                    "model pruning",
                    "knowledge distillation",
                    "quantization",
                ],
            },
        ]

        samples = []
        for i in range(num_samples):
            template = mock_templates[i % len(mock_templates)]
            concept = template["concepts"][i % len(template["concepts"])]

            sample = {
                "input": template["input"].format(concept=concept),
                "output": template["output"].format(concept=concept),
                "source": "instructlab_mock",
                "domain": "machine_learning",
                "generated_at": datetime.now().isoformat(),
            }
            samples.append(sample)

        output_file = (
            self.generated_data_path
            / f"mock_synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"Generated {len(samples)} mock samples saved to {output_file}")
        return output_file

    def _simulate_instructlab_generation(
        self, taxonomy_path: str, num_samples: int, model_name: str
    ) -> List[Dict]:
        """Simulate InstructLab data generation (placeholder for actual implementation)"""
        # This would be replaced with actual InstructLab calls
        with open(taxonomy_path, "r") as f:
            taxonomy = yaml.safe_load(f)

        seed_examples = taxonomy.get("seed_examples", [])
        generated = []

        for i in range(num_samples):
            base_example = seed_examples[i % len(seed_examples)]
            # Add variation to the examples
            generated.append(
                {
                    "question": f"Variation {i+1}: {base_example['question']}",
                    "answer": f"Enhanced answer: {base_example['answer']}",
                    "domain": taxonomy.get("domain", "general"),
                }
            )

        return generated

    def create_knowledge_taxonomy(
        self, domain: str, knowledge_docs: List[str] = None, knowledge_text: str = None
    ) -> Path:
        """Create taxonomy from domain knowledge documents or text

        Args:
            domain: Domain name for the knowledge
            knowledge_docs: List of paths to knowledge documents
            knowledge_text: Direct knowledge text content

        Returns:
            Path to created knowledge taxonomy file
        """
        knowledge_taxonomy = {
            "version": 1,
            "domain": domain,
            "created_by": "QLORAX",
            "description": f"Knowledge taxonomy for {domain} domain",
            "knowledge_sources": [],
            "extracted_concepts": [],
        }

        # Process knowledge documents
        if knowledge_docs:
            for doc_path in knowledge_docs:
                if Path(doc_path).exists():
                    with open(doc_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        knowledge_taxonomy["knowledge_sources"].append(
                            {
                                "document": str(doc_path),
                                "content_preview": (
                                    content[:500] + "..."
                                    if len(content) > 500
                                    else content
                                ),
                            }
                        )

        # Process direct knowledge text
        if knowledge_text:
            knowledge_taxonomy["knowledge_sources"].append(
                {
                    "type": "direct_input",
                    "content": (
                        knowledge_text[:1000] + "..."
                        if len(knowledge_text) > 1000
                        else knowledge_text
                    ),
                }
            )

        # Save knowledge taxonomy
        taxonomy_file = self.knowledge_path / f"{domain}_knowledge.yaml"
        with open(taxonomy_file, "w") as f:
            yaml.dump(knowledge_taxonomy, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created knowledge taxonomy at {taxonomy_file}")
        return taxonomy_file

    def integrate_with_qlorax_training(
        self, synthetic_data_path: str, existing_data_path: str, output_path: str = None
    ) -> Path:
        """Combine synthetic data with existing QLORAX training data

        Args:
            synthetic_data_path: Path to synthetic data file
            existing_data_path: Path to existing training data
            output_path: Output path for combined data

        Returns:
            Path to combined dataset
        """
        combined_data = []

        # Load existing QLORAX data
        if Path(existing_data_path).exists():
            logger.info(f"Loading existing data from {existing_data_path}")
            with open(existing_data_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        sample["source"] = "original_qlorax"
                        combined_data.append(sample)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed JSON at line {line_num}: {e}"
                        )

            logger.info(f"Loaded {len(combined_data)} existing samples")

        # Load synthetic data
        synthetic_count = 0
        if Path(synthetic_data_path).exists():
            logger.info(f"Loading synthetic data from {synthetic_data_path}")
            with open(synthetic_data_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        # Ensure synthetic data has proper source marking
                        if "source" not in sample:
                            sample["source"] = "instructlab_synthetic"
                        combined_data.append(sample)
                        synthetic_count += 1
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed synthetic JSON at line {line_num}: {e}"
                        )

            logger.info(f"Loaded {synthetic_count} synthetic samples")

        # Determine output path
        if not output_path:
            output_path = self.config["output"]["combined_data_file"]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save combined dataset
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in combined_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Log statistics
        original_count = len(combined_data) - synthetic_count
        logger.info(f"Combined dataset statistics:")
        logger.info(f"  Original samples: {original_count}")
        logger.info(f"  Synthetic samples: {synthetic_count}")
        logger.info(f"  Total samples: {len(combined_data)}")
        logger.info(f"  Combined dataset saved to: {output_file}")

        return output_file

    def validate_generated_data(self, data_path: str) -> Dict[str, Any]:
        """Validate the quality of generated synthetic data

        Args:
            data_path: Path to generated data file

        Returns:
            Validation results dictionary
        """
        validation_results = {
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "average_input_length": 0,
            "average_output_length": 0,
            "domains": set(),
            "sources": set(),
            "validation_errors": [],
        }

        if not Path(data_path).exists():
            validation_results["validation_errors"].append(
                f"Data file not found: {data_path}"
            )
            return validation_results

        input_lengths = []
        output_lengths = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                validation_results["total_samples"] += 1

                try:
                    sample = json.loads(line.strip())

                    # Check required fields
                    if "input" not in sample or "output" not in sample:
                        validation_results["invalid_samples"] += 1
                        validation_results["validation_errors"].append(
                            f"Line {line_num}: Missing input or output field"
                        )
                        continue

                    # Check content quality
                    input_text = sample["input"].strip()
                    output_text = sample["output"].strip()

                    if len(input_text) < 10:
                        validation_results["validation_errors"].append(
                            f"Line {line_num}: Input too short (< 10 characters)"
                        )

                    if len(output_text) < 20:
                        validation_results["validation_errors"].append(
                            f"Line {line_num}: Output too short (< 20 characters)"
                        )

                    input_lengths.append(len(input_text))
                    output_lengths.append(len(output_text))

                    # Track metadata
                    if "domain" in sample:
                        validation_results["domains"].add(sample["domain"])
                    if "source" in sample:
                        validation_results["sources"].add(sample["source"])

                    validation_results["valid_samples"] += 1

                except json.JSONDecodeError as e:
                    validation_results["invalid_samples"] += 1
                    validation_results["validation_errors"].append(
                        f"Line {line_num}: JSON decode error - {e}"
                    )

        # Calculate averages
        if input_lengths:
            validation_results["average_input_length"] = sum(input_lengths) / len(
                input_lengths
            )
        if output_lengths:
            validation_results["average_output_length"] = sum(output_lengths) / len(
                output_lengths
            )

        # Convert sets to lists for JSON serialization
        validation_results["domains"] = list(validation_results["domains"])
        validation_results["sources"] = list(validation_results["sources"])

        logger.info(f"Data validation completed:")
        logger.info(f"  Total samples: {validation_results['total_samples']}")
        logger.info(f"  Valid samples: {validation_results['valid_samples']}")
        logger.info(f"  Invalid samples: {validation_results['invalid_samples']}")
        logger.info(
            f"  Average input length: {validation_results['average_input_length']:.1f}"
        )
        logger.info(
            f"  Average output length: {validation_results['average_output_length']:.1f}"
        )

        return validation_results


def main():
    """Demo InstructLab integration functionality"""
    print("🔬 QLORAX InstructLab Integration Demo")
    print("=" * 60)

    try:
        # Initialize integration
        print("\n📋 Step 1: Initializing InstructLab integration...")
        instructlab = QLORAXInstructLab()

        # Check installation
        if not instructlab.check_instructlab_installation():
            print("⚠️  InstructLab not installed. Would install in production.")

        # Create taxonomy
        print("\n📚 Step 2: Creating taxonomy structure...")
        taxonomy_file = instructlab.create_taxonomy_structure("demo_domain")
        print(f"✅ Taxonomy created: {taxonomy_file}")

        # Generate synthetic data
        print("\n🧪 Step 3: Generating synthetic training data...")
        synthetic_data = instructlab.generate_synthetic_data(
            taxonomy_path=str(taxonomy_file), num_samples=25
        )
        print(f"✅ Synthetic data generated: {synthetic_data}")

        # Validate generated data
        print("\n🔍 Step 4: Validating generated data...")
        validation_results = instructlab.validate_generated_data(str(synthetic_data))
        print(
            f"✅ Validation completed: {validation_results['valid_samples']}/{validation_results['total_samples']} valid samples"
        )

        # Integrate with existing data
        existing_data = "data/curated.jsonl"
        if Path(existing_data).exists():
            print(f"\n🔗 Step 5: Combining with existing training data...")
            combined_data = instructlab.integrate_with_qlorax_training(
                synthetic_data_path=str(synthetic_data),
                existing_data_path=existing_data,
                output_path="data/demo_combined.jsonl",
            )
            print(f"✅ Enhanced dataset ready: {combined_data}")
        else:
            print(f"\n⚠️  Existing data not found at {existing_data}")
            print(f"✅ Synthetic dataset ready: {synthetic_data}")

        print("\n🎉 InstructLab integration demo completed successfully!")
        print("\n📋 Next steps:")
        print("1. Install InstructLab: pip install -r requirements-instructlab.txt")
        print("2. Create custom taxonomies for your domain")
        print("3. Generate larger synthetic datasets")
        print("4. Run enhanced training with combined data")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.exception("Demo failed with exception")
        return 1

    return 0


if __name__ == "__main__":
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    exit(main())
