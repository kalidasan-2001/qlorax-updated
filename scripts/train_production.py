#!/usr/bin/env python3
"""
QLORAX Production Training Script
Comprehensive fine-tuning with monitoring, checkpointing, and evaluation
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProductionTrainer:
    """Production-ready QLoRA training with comprehensive monitoring"""

    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_directories()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, "r") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def setup_logging(self):
        """Setup comprehensive logging"""
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File logging
        log_file = self.output_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Training started at {datetime.now()}")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")

    def setup_directories(self):
        """Create necessary directories"""
        dirs = ["checkpoints", "logs", "results", "wandb"]
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(exist_ok=True)

    def setup_wandb(self):
        """Initialize Weights & Biases logging"""
        if self.config.get("use_wandb", False):
            wandb.init(
                project=self.config.get("wandb_project", "qlorax-finetuning"),
                name=self.config.get(
                    "experiment_name", f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                ),
                config=self.config,
                dir=str(self.output_dir / "wandb"),
            )
            logger.info("Weights & Biases initialized")

    def load_and_prepare_data(self) -> Dataset:
        """Load and preprocess training data"""
        logger.info("Loading training data...")

        data_path = self.config["data_path"]
        if data_path.endswith(".jsonl"):
            data = []
            with open(data_path, "r") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif data_path.endswith(".json"):
            with open(data_path, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        logger.info(f"Loaded {len(data)} training examples")

        # Create training prompts
        def format_prompt(example):
            template = self.config.get(
                "prompt_template", "### Input:\n{input}\n\n### Output:\n{output}"
            )
            return template.format(**example)

        formatted_data = [format_prompt(item) for item in data]

        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_data})

        # Split data if validation split specified
        if self.config.get("validation_split", 0) > 0:
            split_ratio = self.config["validation_split"]
            dataset = dataset.train_test_split(test_size=split_ratio, seed=42)
            self.train_dataset = dataset["train"]
            self.eval_dataset = dataset["test"]
            logger.info(
                f"Split data: {len(self.train_dataset)} train, {len(self.eval_dataset)} eval"
            )
        else:
            self.train_dataset = dataset
            self.eval_dataset = None
            logger.info(f"Using {len(self.train_dataset)} examples for training")

        return self.train_dataset

    def load_model_and_tokenizer(self):
        """Load and configure model and tokenizer"""
        logger.info(f"Loading model: {self.config['model_name']}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            trust_remote_code=self.config.get("trust_remote_code", False),
        )

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Added padding token")

        # Load model
        model_kwargs = {
            "torch_dtype": getattr(torch, self.config.get("torch_dtype", "float16")),
            "device_map": self.config.get("device_map", "auto"),
            "trust_remote_code": self.config.get("trust_remote_code", False),
        }

        # Add quantization config if specified
        if self.config.get("load_in_4bit", False):
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=self.config.get(
                    "bnb_4bit_use_double_quant", True
                ),
                bnb_4bit_compute_dtype=getattr(
                    torch, self.config.get("bnb_4bit_compute_dtype", "bfloat16")
                ),
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"], **model_kwargs
        )

        logger.info(f"Model loaded with {self.model.num_parameters():,} parameters")

    def setup_lora(self):
        """Configure and apply LoRA"""
        logger.info("Setting up LoRA...")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.get("lora_r", 32),
            lora_alpha=self.config.get("lora_alpha", 64),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            target_modules=self.config.get(
                "lora_target_modules",
                [
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "down_proj",
                    "up_proj",
                ],
            ),
            bias=self.config.get("lora_bias", "none"),
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)"
        )

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset"""
        logger.info("Tokenizing dataset...")

        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # Let data collator handle padding
                max_length=self.config.get("max_length", 1024),
                return_tensors=None,  # Return lists, not tensors
            )
            # For causal language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

        logger.info(f"Tokenized {len(tokenized)} examples")
        return tokenized

    def create_trainer(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ) -> Trainer:
        """Create and configure trainer"""
        logger.info("Creating trainer...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            overwrite_output_dir=True,
            # Training schedule
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get(
                "per_device_train_batch_size", 1
            ),
            per_device_eval_batch_size=self.config.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=self.config.get(
                "gradient_accumulation_steps", 4
            ),
            # Optimization
            learning_rate=self.config.get("learning_rate", 2e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            adam_beta1=self.config.get("adam_beta1", 0.9),
            adam_beta2=self.config.get("adam_beta2", 0.999),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            # Schedule
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
            warmup_ratio=self.config.get("warmup_ratio", 0.1),
            # Logging and saving
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 500),
            save_total_limit=self.config.get("save_total_limit", 3),
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.get("eval_steps", 500) if eval_dataset else None,
            # Performance
            fp16=self.config.get("fp16", False),
            bf16=self.config.get("bf16", False),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            dataloader_num_workers=self.config.get("dataloader_num_workers", 0),
            # Monitoring - Use empty list instead of None to properly disable wandb
            report_to=["wandb"] if self.config.get("use_wandb", False) else [],
            run_name=self.config.get("experiment_name"),
            # Misc
            remove_unused_columns=False,
            load_best_model_at_end=True if eval_dataset else False,
            seed=self.config.get("seed", 42),
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Callbacks
        callbacks = []
        if eval_dataset and self.config.get("early_stopping_patience"):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config["early_stopping_patience"]
                )
            )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        return trainer

    def train(self):
        """Run the complete training pipeline"""
        try:
            logger.info("Starting production training pipeline...")

            # Setup monitoring
            self.setup_wandb()

            # Load and prepare data
            self.load_and_prepare_data()

            # Load model and tokenizer
            self.load_model_and_tokenizer()

            # Setup LoRA
            self.setup_lora()

            # Tokenize datasets
            tokenized_train = self.tokenize_dataset(self.train_dataset)
            tokenized_eval = (
                self.tokenize_dataset(self.eval_dataset) if self.eval_dataset else None
            )

            # Create trainer
            trainer = self.create_trainer(tokenized_train, tokenized_eval)

            # Start training
            logger.info("Starting training...")
            train_result = trainer.train()

            # Save the model
            logger.info("Saving model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)

            # Save training results
            results = {
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics[
                    "train_samples_per_second"
                ],
                "total_flos": train_result.metrics.get("total_flos", 0),
                "config": self.config,
            }

            with open(self.output_dir / "results" / "training_results.json", "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Training completed successfully!")
            logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            logger.info(f"Model saved to: {self.output_dir}")

            # Close wandb
            if self.config.get("use_wandb", False):
                wandb.finish()

            return results

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            if self.config.get("use_wandb", False):
                wandb.finish(exit_code=1)
            raise e


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="QLORAX Production Training")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--data", help="Override data path from config")
    parser.add_argument("--output", help="Override output directory from config")
    parser.add_argument("--wandb-project", help="Weights & Biases project name")
    parser.add_argument("--experiment-name", help="Experiment name for logging")

    args = parser.parse_args()

    # Initialize trainer
    trainer = ProductionTrainer(args.config)

    # Override config with command line arguments
    if args.data:
        trainer.config["data_path"] = args.data
    if args.output:
        trainer.config["output_dir"] = args.output
    if args.wandb_project:
        trainer.config["wandb_project"] = args.wandb_project
        trainer.config["use_wandb"] = True
    if args.experiment_name:
        trainer.config["experiment_name"] = args.experiment_name

    # Run training
    results = trainer.train()

    print("\n🎉 Training completed successfully!")
    print(f"📁 Model saved to: {trainer.output_dir}")
    print(f"📊 Final loss: {results['train_loss']:.4f}")
    print(f"⏱️  Training time: {results['train_runtime']:.2f} seconds")
    print(f"🚀 Samples per second: {results['train_samples_per_second']:.2f}")


if __name__ == "__main__":
    main()
