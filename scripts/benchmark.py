#!/usr/bin/env python3
"""
QLORAX Comprehensive Benchmarking & Evaluation Suite
Evaluates fine-tuned models across multiple metrics and generates detailed reports
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftModel
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class ModelBenchmark:
    """Comprehensive model evaluation and benchmarking"""

    def __init__(self, model_path: str, test_data_path: str, output_dir: str):
        """Initialize benchmark suite"""
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}
        self.predictions = []

        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.semantic_model = None

        print(f"🔬 Initializing benchmark for model: {self.model_path}")
        print(f"📊 Test data: {self.test_data_path}")
        print(f"📁 Results will be saved to: {self.output_dir}")

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print("📥 Loading model and tokenizer...")

        # Determine if this is a PEFT model or full model
        if (self.model_path / "adapter_config.json").exists():
            # Load PEFT model
            adapter_config_path = self.model_path / "adapter_config.json"
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)

            base_model_name = adapter_config.get("base_model_name_or_path")
            if not base_model_name:
                raise ValueError("Cannot find base model name in adapter config")

            print(f"📦 Loading base model: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.float32, device_map="cpu"
            )

            print(f"🔗 Loading LoRA adapter from: {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
        else:
            # Load full model
            print(f"📦 Loading full model from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), torch_dtype=torch.float32, device_map="cpu"
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print("✅ Model loaded successfully")

    def load_test_data(self) -> List[Dict[str, str]]:
        """Load test dataset"""
        print("📊 Loading test data...")

        test_data = []
        if self.test_data_path.suffix == ".jsonl":
            with open(self.test_data_path, "r") as f:
                for line in f:
                    test_data.append(json.loads(line.strip()))
        elif self.test_data_path.suffix == ".json":
            with open(self.test_data_path, "r") as f:
                test_data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {self.test_data_path.suffix}")

        print(f"📋 Loaded {len(test_data)} test examples")
        return test_data

    def generate_prediction(self, input_text: str, max_length: int = 512) -> str:
        """Generate prediction for a single input"""
        prompt = f"### Input:\n{input_text}\n\n### Output:\n"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part
        response = generated_text[len(prompt) :].strip()
        return response

    def calculate_perplexity(self, test_data: List[Dict[str, str]]) -> float:
        """Calculate perplexity on test set"""
        print("📈 Calculating perplexity...")

        total_loss = 0
        total_tokens = 0

        for example in tqdm(test_data, desc="Computing perplexity"):
            prompt = (
                f"### Input:\n{example['input']}\n\n### Output:\n{example['output']}"
            )
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        print(f"📊 Perplexity: {perplexity:.2f}")
        return perplexity

    def calculate_bleu_scores(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Calculate BLEU scores"""
        print("🔤 Calculating BLEU scores...")

        smoothing = SmoothingFunction().method1
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_4_scores = []

        for pred, ref in tqdm(
            zip(predictions, references), desc="Computing BLEU", total=len(predictions)
        ):
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = [nltk.word_tokenize(ref.lower())]

            # BLEU-1
            bleu_1 = sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=(1, 0, 0, 0),
                smoothing_function=smoothing,
            )
            bleu_1_scores.append(bleu_1)

            # BLEU-2
            bleu_2 = sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=smoothing,
            )
            bleu_2_scores.append(bleu_2)

            # BLEU-4
            bleu_4 = sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )
            bleu_4_scores.append(bleu_4)

        results = {
            "bleu_1": np.mean(bleu_1_scores) * 100,
            "bleu_2": np.mean(bleu_2_scores) * 100,
            "bleu_4": np.mean(bleu_4_scores) * 100,
        }

        print(f"📊 BLEU-1: {results['bleu_1']:.2f}")
        print(f"📊 BLEU-2: {results['bleu_2']:.2f}")
        print(f"📊 BLEU-4: {results['bleu_4']:.2f}")

        return results

    def calculate_rouge_scores(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        print("🌹 Calculating ROUGE scores...")

        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []

        for pred, ref in tqdm(
            zip(predictions, references), desc="Computing ROUGE", total=len(predictions)
        ):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_1_scores.append(scores["rouge1"].fmeasure)
            rouge_2_scores.append(scores["rouge2"].fmeasure)
            rouge_l_scores.append(scores["rougeL"].fmeasure)

        results = {
            "rouge_1": np.mean(rouge_1_scores),
            "rouge_2": np.mean(rouge_2_scores),
            "rouge_l": np.mean(rouge_l_scores),
        }

        print(f"📊 ROUGE-1: {results['rouge_1']:.3f}")
        print(f"📊 ROUGE-2: {results['rouge_2']:.3f}")
        print(f"📊 ROUGE-L: {results['rouge_l']:.3f}")

        return results

    def calculate_semantic_similarity(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Calculate semantic similarity using sentence transformers"""
        print("🧠 Calculating semantic similarity...")

        if self.semantic_model is None:
            print("📥 Loading sentence transformer model...")
            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Encode sentences
        pred_embeddings = self.semantic_model.encode(predictions)
        ref_embeddings = self.semantic_model.encode(references)

        # Calculate cosine similarities
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            similarity = cosine_similarity([pred_emb], [ref_emb])[0][0]
            similarities.append(similarity)

        avg_similarity = np.mean(similarities)
        print(f"📊 Semantic Similarity: {avg_similarity:.3f}")

        return avg_similarity

    def calculate_exact_match(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Calculate exact match accuracy"""
        print("🎯 Calculating exact match...")

        exact_matches = 0
        for pred, ref in zip(predictions, references):
            if pred.strip().lower() == ref.strip().lower():
                exact_matches += 1

        exact_match_ratio = exact_matches / len(predictions)
        print(f"📊 Exact Match: {exact_match_ratio:.3f}")

        return exact_match_ratio

    def measure_performance(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """Measure inference performance"""
        print("⚡ Measuring performance...")

        # Warm up
        for _ in range(3):
            self.generate_prediction("Test input")

        # Measure inference time
        start_time = time.time()
        sample_size = min(50, len(test_data))

        for i in range(sample_size):
            self.generate_prediction(test_data[i]["input"])

        end_time = time.time()

        avg_inference_time = (
            (end_time - start_time) / sample_size * 1000
        )  # milliseconds
        throughput = 1000 / avg_inference_time  # samples per second

        # Estimate memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            memory_used = 0  # CPU memory measurement is more complex

        performance = {
            "avg_inference_time_ms": avg_inference_time,
            "throughput_samples_per_sec": throughput,
            "memory_usage_mb": memory_used,
        }

        print(f"📊 Avg Inference Time: {avg_inference_time:.2f} ms")
        print(f"📊 Throughput: {throughput:.2f} samples/sec")
        print(f"📊 Memory Usage: {memory_used:.2f} MB")

        return performance

    def generate_predictions(self, test_data: List[Dict[str, str]]) -> List[str]:
        """Generate predictions for all test examples"""
        print("🤖 Generating predictions...")

        predictions = []
        for example in tqdm(test_data, desc="Generating"):
            prediction = self.generate_prediction(example["input"])
            predictions.append(prediction)

            # Store for detailed analysis
            self.predictions.append(
                {
                    "input": example["input"],
                    "reference": example["output"],
                    "prediction": prediction,
                }
            )

        return predictions

    def create_visualizations(self):
        """Create visualization plots"""
        print("📊 Creating visualizations...")

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Model Evaluation Results", fontsize=16, fontweight="bold")

        # 1. Metrics overview
        metrics = ["Perplexity", "BLEU-4", "ROUGE-L", "Semantic Sim"]
        values = [
            self.results.get("perplexity", 0),
            self.results.get("bleu_4", 0),
            self.results.get("rouge_l", 0) * 100,  # Scale to 0-100
            self.results.get("semantic_similarity", 0) * 100,  # Scale to 0-100
        ]

        axes[0, 0].bar(
            metrics, values, color=["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4"]
        )
        axes[0, 0].set_title("Key Metrics Overview")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. BLEU scores
        bleu_metrics = ["BLEU-1", "BLEU-2", "BLEU-4"]
        bleu_values = [
            self.results.get("bleu_1", 0),
            self.results.get("bleu_2", 0),
            self.results.get("bleu_4", 0),
        ]

        axes[0, 1].plot(
            bleu_metrics, bleu_values, marker="o", linewidth=2, markersize=8
        )
        axes[0, 1].set_title("BLEU Scores")
        axes[0, 1].set_ylabel("BLEU Score")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. ROUGE scores
        rouge_metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
        rouge_values = [
            self.results.get("rouge_1", 0),
            self.results.get("rouge_2", 0),
            self.results.get("rouge_l", 0),
        ]

        axes[1, 0].bar(
            rouge_metrics, rouge_values, color=["#ffa07a", "#98d8c8", "#f7dc6f"]
        )
        axes[1, 0].set_title("ROUGE Scores")
        axes[1, 0].set_ylabel("ROUGE Score")

        # 4. Performance metrics
        perf_metrics = ["Inference Time (ms)", "Throughput (samples/s)"]
        perf_values = [
            self.results.get("avg_inference_time_ms", 0),
            self.results.get("throughput_samples_per_sec", 0),
        ]

        # Use secondary y-axis for different scales
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()

        bar1 = ax1.bar(["Inference Time"], [perf_values[0]], color="#ff6b6b", alpha=0.7)
        bar2 = ax2.bar(["Throughput"], [perf_values[1]], color="#4ecdc4", alpha=0.7)

        ax1.set_ylabel("Time (ms)", color="#ff6b6b")
        ax2.set_ylabel("Samples/sec", color="#4ecdc4")
        axes[1, 1].set_title("Performance Metrics")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "evaluation_results.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(
            f"📊 Visualizations saved to: {self.output_dir / 'evaluation_results.png'}"
        )

    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("📝 Generating evaluation report...")

        report = f"""
# Model Evaluation Report

**Model:** {self.model_path}
**Test Data:** {self.test_data_path}
**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report presents a comprehensive evaluation of the fine-tuned model across multiple metrics.

## Metrics Overview

### Language Modeling Quality
- **Perplexity:** {self.results.get('perplexity', 'N/A')} (lower is better)

### Generation Quality
- **BLEU-1:** {self.results.get('bleu_1', 'N/A'):.2f}
- **BLEU-2:** {self.results.get('bleu_2', 'N/A'):.2f}
- **BLEU-4:** {self.results.get('bleu_4', 'N/A'):.2f}

### Semantic Quality
- **ROUGE-1:** {self.results.get('rouge_1', 'N/A'):.3f}
- **ROUGE-2:** {self.results.get('rouge_2', 'N/A'):.3f}
- **ROUGE-L:** {self.results.get('rouge_l', 'N/A'):.3f}
- **Semantic Similarity:** {self.results.get('semantic_similarity', 'N/A'):.3f}

### Accuracy
- **Exact Match:** {self.results.get('exact_match', 'N/A'):.3f}

### Performance
- **Average Inference Time:** {self.results.get('avg_inference_time_ms', 'N/A'):.2f} ms
- **Throughput:** {self.results.get('throughput_samples_per_sec', 'N/A'):.2f} samples/sec
- **Memory Usage:** {self.results.get('memory_usage_mb', 'N/A'):.2f} MB

## Interpretation

### Perplexity
- Score: {self.results.get('perplexity', 'N/A')}
- Interpretation: {"Excellent (< 5)" if self.results.get('perplexity', float('inf')) < 5 else "Good (< 10)" if self.results.get('perplexity', float('inf')) < 10 else "Needs Improvement (≥ 10)"}

### BLEU-4 Score
- Score: {self.results.get('bleu_4', 'N/A'):.2f}
- Interpretation: {"Excellent (≥ 40)" if self.results.get('bleu_4', 0) >= 40 else "Good (≥ 20)" if self.results.get('bleu_4', 0) >= 20 else "Needs Improvement (< 20)"}

### ROUGE-L Score
- Score: {self.results.get('rouge_l', 'N/A'):.3f}
- Interpretation: {"Excellent (≥ 0.5)" if self.results.get('rouge_l', 0) >= 0.5 else "Good (≥ 0.3)" if self.results.get('rouge_l', 0) >= 0.3 else "Needs Improvement (< 0.3)"}

## Recommendations

### Training Improvements
"""

        # Add recommendations based on scores
        if self.results.get("perplexity", float("inf")) > 10:
            report += "\n- Consider increasing training epochs or improving data quality to reduce perplexity"

        if self.results.get("bleu_4", 0) < 20:
            report += "\n- BLEU scores could be improved with more training data or better prompt engineering"

        if self.results.get("rouge_l", 0) < 0.3:
            report += "\n- ROUGE scores suggest the model could benefit from longer training or better target modules"

        report += f"""

### Deployment Considerations
- **Inference Speed:** {"Fast" if self.results.get('avg_inference_time_ms', float('inf')) < 100 else "Moderate" if self.results.get('avg_inference_time_ms', float('inf')) < 500 else "Slow"} ({self.results.get('avg_inference_time_ms', 'N/A'):.2f} ms per sample)
- **Memory Efficiency:** {"Efficient" if self.results.get('memory_usage_mb', 0) < 2048 else "Moderate" if self.results.get('memory_usage_mb', 0) < 8192 else "Memory Intensive"}

## Example Predictions

"""

        # Add sample predictions
        for i, pred in enumerate(self.predictions[:3]):
            report += f"""
### Example {i+1}
**Input:** {pred['input']}
**Reference:** {pred['reference']}
**Prediction:** {pred['prediction']}
---
"""

        report += "\n## Files Generated\n"
        report += f"- Detailed results: `{self.output_dir / 'detailed_results.json'}`\n"
        report += f"- Predictions: `{self.output_dir / 'predictions.json'}`\n"
        report += f"- Visualizations: `{self.output_dir / 'evaluation_results.png'}`\n"

        # Save report
        with open(self.output_dir / "evaluation_report.md", "w") as f:
            f.write(report)

        print(f"📄 Report saved to: {self.output_dir / 'evaluation_report.md'}")

    def run_benchmark(self):
        """Run complete benchmark suite"""
        print("🚀 Starting comprehensive benchmark...")
        start_time = time.time()

        # Load model and data
        self.load_model()
        test_data = self.load_test_data()

        # Generate predictions
        predictions = self.generate_predictions(test_data)
        references = [item["output"] for item in test_data]

        # Calculate all metrics
        self.results["perplexity"] = self.calculate_perplexity(test_data)
        bleu_scores = self.calculate_bleu_scores(predictions, references)
        rouge_scores = self.calculate_rouge_scores(predictions, references)
        self.results["semantic_similarity"] = self.calculate_semantic_similarity(
            predictions, references
        )
        self.results["exact_match"] = self.calculate_exact_match(
            predictions, references
        )
        performance = self.measure_performance(test_data)

        # Combine all results
        self.results.update(bleu_scores)
        self.results.update(rouge_scores)
        self.results.update(performance)

        # Add metadata
        self.results["model_path"] = str(self.model_path)
        self.results["test_data_path"] = str(self.test_data_path)
        self.results["num_test_examples"] = len(test_data)
        self.results["evaluation_time"] = time.time() - start_time
        self.results["timestamp"] = datetime.now().isoformat()

        # Save detailed results
        with open(self.output_dir / "detailed_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        # Save predictions
        with open(self.output_dir / "predictions.json", "w") as f:
            json.dump(self.predictions, f, indent=2)

        # Create visualizations
        self.create_visualizations()

        # Generate report
        self.generate_report()

        print(
            f"\n🎉 Benchmark completed in {self.results['evaluation_time']:.2f} seconds!"
        )
        print(f"📁 All results saved to: {self.output_dir}")

        return self.results


def main():
    """Main benchmarking function"""
    parser = argparse.ArgumentParser(description="QLORAX Model Benchmarking")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--test-data", required=True, help="Path to test dataset")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick evaluation (subset of metrics)"
    )

    args = parser.parse_args()

    # Initialize and run benchmark
    benchmark = ModelBenchmark(args.model, args.test_data, args.output)
    results = benchmark.run_benchmark()

    # Print summary
    print("\n📊 BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Perplexity: {results.get('perplexity', 'N/A'):.2f}")
    print(f"BLEU-4: {results.get('bleu_4', 'N/A'):.2f}")
    print(f"ROUGE-L: {results.get('rouge_l', 'N/A'):.3f}")
    print(f"Semantic Similarity: {results.get('semantic_similarity', 'N/A'):.3f}")
    print(f"Exact Match: {results.get('exact_match', 'N/A'):.3f}")
    print(f"Avg Inference Time: {results.get('avg_inference_time_ms', 'N/A'):.2f} ms")


if __name__ == "__main__":
    main()
