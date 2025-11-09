#!/usr/bin/env python3
"""
🤗 QLORAX Hugging Face Hub Integration
Automated model publishing and artifact management for CI/CD pipeline
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from huggingface_hub import HfApi, create_repo, hf_hub_download, upload_folder
    from huggingface_hub.utils import RepositoryNotFoundError

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning(
        "huggingface_hub not available. Install with: pip install huggingface_hub"
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HuggingFacePublisher:
    """Hugging Face Hub integration for QLORAX model publishing"""

    def __init__(self, token: Optional[str] = None, organization: Optional[str] = None):
        """Initialize Hugging Face publisher

        Args:
            token: HuggingFace API token (or set HF_TOKEN environment variable)
            organization: Organization name for model repositories
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )

        self.token = token or os.getenv("HF_TOKEN")
        self.organization = organization or os.getenv("HF_ORGANIZATION")

        if not self.token:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN environment variable or pass token parameter."
            )

        self.api = HfApi(token=self.token)
        logger.info("HuggingFace Hub integration initialized")

    def create_model_repository(
        self, repo_name: str, private: bool = False, description: Optional[str] = None
    ) -> str:
        """Create a model repository on Hugging Face Hub

        Args:
            repo_name: Repository name
            private: Whether to create private repository
            description: Repository description

        Returns:
            Full repository ID (organization/repo_name or username/repo_name)
        """
        try:
            # Construct full repository ID
            if self.organization:
                repo_id = f"{self.organization}/{repo_name}"
            else:
                # Use username if no organization specified
                user_info = self.api.whoami()
                username = user_info["name"]
                repo_id = f"{username}/{repo_name}"

            # Check if repository already exists
            try:
                repo_info = self.api.repo_info(repo_id, repo_type="model")
                logger.info(f"Repository {repo_id} already exists")
                return repo_id
            except RepositoryNotFoundError:
                pass

            # Create repository
            logger.info(f"Creating repository: {repo_id}")
            self.api.create_repo(
                repo_id=repo_id, repo_type="model", private=private, exist_ok=True
            )

            # Update repository description if provided
            if description:
                self.api.update_repo_visibility(repo_id, private=private)

            logger.info(f"✅ Repository created successfully: {repo_id}")
            return repo_id

        except Exception as e:
            logger.error(f"Failed to create repository {repo_name}: {e}")
            raise

    def publish_model(
        self,
        model_path: str,
        repo_name: str,
        commit_message: Optional[str] = None,
        private: bool = False,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Publish trained model to Hugging Face Hub

        Args:
            model_path: Path to trained model directory
            repo_name: Repository name for the model
            commit_message: Commit message for the upload
            private: Whether repository should be private
            model_metadata: Additional metadata for the model

        Returns:
            Repository URL
        """
        model_dir = Path(model_path)
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # Create repository
        repo_id = self.create_model_repository(repo_name, private=private)

        # Generate commit message
        if not commit_message:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"QLORAX automated model upload - {timestamp}"

        # Create model card
        model_card_path = model_dir / "README.md"
        if not model_card_path.exists():
            self.create_model_card(model_dir, model_metadata or {})

        try:
            # Upload model files
            logger.info(f"📤 Uploading model from {model_path} to {repo_id}")

            self.api.upload_folder(
                folder_path=str(model_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                ignore_patterns=["*.git*", "*.pyc", "__pycache__", ".DS_Store"],
            )

            # Generate repository URL
            repo_url = f"https://huggingface.co/{repo_id}"
            logger.info(f"✅ Model published successfully: {repo_url}")

            return repo_url

        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            raise

    def create_model_card(self, model_dir: Path, metadata: Dict[str, Any]):
        """Create a model card (README.md) for the repository

        Args:
            model_dir: Model directory path
            metadata: Model metadata
        """
        model_card_content = self.generate_model_card_content(metadata)

        model_card_path = model_dir / "README.md"
        with open(model_card_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)

        logger.info(f"Model card created: {model_card_path}")

    def generate_model_card_content(self, metadata: Dict[str, Any]) -> str:
        """Generate model card content

        Args:
            metadata: Model metadata

        Returns:
            Model card content as markdown string
        """
        # Extract metadata
        model_name = metadata.get("model_name", "QLORAX Enhanced Model")
        base_model = metadata.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        training_data = metadata.get(
            "training_data", "Synthetic data generated with InstructLab"
        )
        performance_metrics = metadata.get("performance_metrics", {})
        training_params = metadata.get("training_parameters", {})

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d")

        model_card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- qlorax
- qlora
- fine-tuned
- instructlab
- synthetic-data
- llm
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

# {model_name}

This model is a QLoRA (Quantized Low-Rank Adaptation) fine-tuned version of `{base_model}`, enhanced with synthetic data generated using InstructLab integration.

## 🎯 Model Overview

- **Base Model:** {base_model}
- **Fine-tuning Method:** QLoRA (4-bit quantization + LoRA adapters)
- **Training Data:** {training_data}
- **Generated:** {timestamp}
- **Framework:** QLORAX Enhanced Training Pipeline

## 📊 Performance Metrics

"""

        # Add performance metrics if available
        if performance_metrics:
            model_card += "| Metric | Score |\n|--------|-------|\n"
            for metric, score in performance_metrics.items():
                if isinstance(score, float):
                    model_card += (
                        f"| {metric.replace('_', ' ').title()} | {score:.4f} |\n"
                    )
                else:
                    model_card += f"| {metric.replace('_', ' ').title()} | {score} |\n"
            model_card += "\n"

        # Add training parameters
        if training_params:
            model_card += "## 🔧 Training Parameters\n\n"
            for param, value in training_params.items():
                model_card += f"- **{param.replace('_', ' ').title()}:** {value}\n"
            model_card += "\n"

        # Add usage section
        model_card += """## 🚀 Usage

### Using with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "path/to/this/model")

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using with QLORAX

```python
from scripts.test_model import load_enhanced_model

# Load model using QLORAX utilities
model, tokenizer = load_enhanced_model("path/to/this/model")

# Interactive testing
from scripts.gradio_app import create_interface
interface = create_interface(model, tokenizer)
interface.launch()
```

## 📈 Evaluation

This model has been evaluated using the QLORAX enhanced benchmarking suite, which includes:

- **ROUGE Scores:** Text similarity and overlap metrics
- **BERT Scores:** Semantic similarity using BERT embeddings  
- **Coherence Metrics:** Response quality and logical consistency
- **Domain-Specific Evaluation:** Task-specific performance metrics

## 🔬 Technical Details

### QLoRA Configuration

- **Rank (r):** 8
- **Alpha:** 16  
- **Dropout:** 0.1
- **Target Modules:** q_proj, v_proj, k_proj, o_proj, gate_proj, down_proj, up_proj
- **Quantization:** 4-bit NF4 with double quantization

### InstructLab Integration

This model benefits from synthetic data generation using InstructLab, which provides:

- Domain-specific knowledge injection
- High-quality question-answer pair generation
- Taxonomy-based data organization
- Automated data validation and quality control

## 🏗️ QLORAX Framework

This model was trained using the QLORAX (QLoRA eXtended) framework, which provides:

- **Automated CI/CD Pipeline:** Continuous integration and deployment for model training
- **Advanced Benchmarking:** Comprehensive evaluation with multiple metrics
- **Production Deployment:** Ready-to-use serving containers and APIs
- **Quality Gates:** Automated quality control and validation

## 📄 License

This model is released under the Apache 2.0 License. Please refer to the original base model license for any additional restrictions.

## 🤝 Citation

If you use this model in your research, please cite:

```bibtex
@misc{{qlorax_enhanced_model_{timestamp.replace('-', '_')},
  title={{QLORAX Enhanced Model: QLoRA Fine-tuning with InstructLab Integration}},
  author={{QLORAX Team}},
  year={{{timestamp[:4]}}},
  url={{https://huggingface.co/your-org/this-model}}
}}
```

## 🔗 Links

- **QLORAX Repository:** [GitHub](https://github.com/your-org/qlorax)
- **Base Model:** [{base_model}](https://huggingface.co/{base_model})
- **InstructLab:** [GitHub](https://github.com/instructlab/instructlab)

---

*Generated automatically by QLORAX Enhanced Training Pipeline*
""".format(
            base_model=base_model, timestamp=timestamp
        )

        return model_card

    def download_model(
        self, repo_id: str, local_dir: str, revision: str = "main"
    ) -> str:
        """Download model from Hugging Face Hub

        Args:
            repo_id: Repository ID (organization/model_name)
            local_dir: Local directory to download to
            revision: Git revision to download

        Returns:
            Path to downloaded model directory
        """
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"📥 Downloading model {repo_id} to {local_dir}")

            # Download all files from the repository
            self.api.snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(local_path),
                revision=revision,
            )

            logger.info(f"✅ Model downloaded successfully to: {local_path}")
            return str(local_path)

        except Exception as e:
            logger.error(f"Failed to download model {repo_id}: {e}")
            raise

    def list_models(self, organization: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models in organization or user account

        Args:
            organization: Organization to list models from

        Returns:
            List of model information
        """
        try:
            if organization or self.organization:
                org = organization or self.organization
                models = self.api.list_models(author=org)
            else:
                # List user's models
                user_info = self.api.whoami()
                username = user_info["name"]
                models = self.api.list_models(author=username)

            model_list = []
            for model in models:
                model_info = {
                    "id": model.modelId,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "created_at": (
                        model.createdAt.isoformat() if model.createdAt else None
                    ),
                    "updated_at": (
                        model.lastModified.isoformat() if model.lastModified else None
                    ),
                }
                model_list.append(model_info)

            return model_list

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise


def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description="QLORAX Hugging Face Hub Integration")
    parser.add_argument(
        "action", choices=["publish", "download", "list"], help="Action to perform"
    )
    parser.add_argument("--model-path", help="Path to model directory (for publish)")
    parser.add_argument("--repo-name", help="Repository name")
    parser.add_argument("--repo-id", help="Repository ID (for download)")
    parser.add_argument("--local-dir", help="Local directory (for download)")
    parser.add_argument(
        "--private", action="store_true", help="Create private repository"
    )
    parser.add_argument("--token", help="HuggingFace API token")
    parser.add_argument("--organization", help="Organization name")
    parser.add_argument("--commit-message", help="Commit message for upload")
    parser.add_argument("--metadata-file", help="Path to metadata JSON file")

    args = parser.parse_args()

    # Initialize publisher
    publisher = HuggingFacePublisher(token=args.token, organization=args.organization)

    if args.action == "publish":
        if not args.model_path or not args.repo_name:
            raise ValueError("--model-path and --repo-name required for publish action")

        # Load metadata if provided
        metadata = {}
        if args.metadata_file and Path(args.metadata_file).exists():
            with open(args.metadata_file, "r") as f:
                metadata = json.load(f)

        # Publish model
        repo_url = publisher.publish_model(
            model_path=args.model_path,
            repo_name=args.repo_name,
            commit_message=args.commit_message,
            private=args.private,
            model_metadata=metadata,
        )

        print(f"✅ Model published successfully: {repo_url}")

    elif args.action == "download":
        if not args.repo_id or not args.local_dir:
            raise ValueError("--repo-id and --local-dir required for download action")

        # Download model
        local_path = publisher.download_model(
            repo_id=args.repo_id, local_dir=args.local_dir
        )

        print(f"✅ Model downloaded to: {local_path}")

    elif args.action == "list":
        # List models
        models = publisher.list_models(args.organization)

        print(f"📋 Found {len(models)} models:")
        for model in models:
            print(f"   📄 {model['id']} (↓{model['downloads']} ❤️{model['likes']})")


if __name__ == "__main__":
    main()
