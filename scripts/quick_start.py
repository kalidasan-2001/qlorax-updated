#!/usr/bin/env python3
"""
QLORAX Enhanced Quick Start Script
Complete fine-tuning and benchmarking pipeline with InstructLab integration
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(command, description, check=True):
    """Run a command with proper error handling"""
    print(f"🚀 {description}")
    print(f"📋 Command: {command}")

    try:
        if isinstance(command, str):
            result = subprocess.run(
                command, shell=True, check=check, capture_output=True, text=True
            )
        else:
            result = subprocess.run(
                command, check=check, capture_output=True, text=True
            )

        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print(f"📄 Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed")
            if result.stderr:
                print(f"❌ Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

    return True


def validate_setup():
    """Validate the environment setup"""
    print("\n🔍 Validating setup...")

    # Check if we're in the right directory
    if not Path("scripts/train_production.py").exists():
        print("❌ Please run this script from the QLORAX project root directory")
        return False

    # Check if virtual environment is activated
    python_path = sys.executable
    if "venv" not in python_path:
        print("⚠️  Virtual environment may not be activated")
        print(f"   Python path: {python_path}")

    # Check key files
    required_files = [
        "scripts/train_production.py",
        "scripts/benchmark.py",
        "configs/production-config.yaml",
        "data/training_data.jsonl",
        "data/test_data.jsonl",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    print("✅ Setup validation passed")
    return True


def run_training():
    """Run production training with fallback to test config"""
    print("\n🎯 Starting Production Training")
    print("=" * 50)

    # Try production config first
    command = [
        sys.executable,
        "scripts/train_production.py",
        "--config",
        "configs/production-config.yaml",
    ]

    success = run_command(command, "Production training", check=False)

    if not success:
        print("\n⚠️  Production training failed. Trying with test configuration...")
        print("   (Using smaller model for compatibility)")

        # Fallback to test config
        command_test = [
            sys.executable,
            "scripts/train_production.py",
            "--config",
            "configs/test-config.yaml",
            "--output",
            "models/production-model",  # Use same output path
        ]

        success = run_command(
            command_test, "Fallback training (test config)", check=False
        )

        if success:
            print("✅ Training completed successfully with test configuration!")
            print("💡 The model is smaller but fully functional for benchmarking.")

    return success


def run_benchmark(model_path):
    """Run comprehensive benchmarking"""
    print("\n📊 Starting Comprehensive Benchmarking")
    print("=" * 50)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/benchmark_{timestamp}"

    command = [
        sys.executable,
        "scripts/benchmark.py",
        "--model",
        model_path,
        "--test-data",
        "data/test_data.jsonl",
        "--output",
        results_dir,
    ]

    success = run_command(command, "Model benchmarking", check=False)

    if success:
        print(f"📁 Results saved to: {results_dir}")
        return results_dir

    return None


def show_results(results_dir, enhanced=False):
    """Display benchmark results"""
    if not results_dir or not Path(results_dir).exists():
        print("❌ No results to display")
        return

    # Look for enhanced results first, then fall back to standard
    enhanced_results_files = list(
        Path(results_dir).glob("enhanced_benchmark_results_*.json")
    )
    standard_results_file = Path(results_dir) / "detailed_results.json"

    results_file = None
    if enhanced and enhanced_results_files:
        results_file = max(enhanced_results_files, key=lambda p: p.stat().st_mtime)
    elif standard_results_file.exists():
        results_file = standard_results_file

    if not results_file or not results_file.exists():
        print("❌ No results file found")
        return

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        if enhanced and "instructlab_metrics" in results:
            print("\n� ENHANCED BENCHMARK RESULTS WITH INSTRUCTLAB")
            print("=" * 60)

            # Standard metrics
            metrics = results.get("metrics", {})
            rouge_scores = metrics.get("rouge_scores", {})
            bert_scores = metrics.get("bert_scores", {})
            quality_metrics = metrics.get("quality_metrics", {})

            print("�📊 Standard Metrics:")
            print(f"   🌹 ROUGE-L: {rouge_scores.get('rougeL', 0):.4f}")
            print(f"   🌹 ROUGE-1: {rouge_scores.get('rouge1', 0):.4f}")
            print(f"   🌹 ROUGE-2: {rouge_scores.get('rouge2', 0):.4f}")
            print(f"   🧠 BERT F1: {bert_scores.get('bert_f1', 0):.4f}")
            print(
                f"   📏 Avg Response Length: {quality_metrics.get('average_response_length', 0):.1f}"
            )
            print(
                f"   🌈 Response Diversity: {quality_metrics.get('response_diversity', 0):.4f}"
            )

            # InstructLab metrics
            instructlab_metrics = results.get("instructlab_metrics", {})

            print("\n🔬 InstructLab Enhancement Metrics:")

            # Synthetic data impact
            synthetic_impact = instructlab_metrics.get("synthetic_data_impact", {})
            if synthetic_impact:
                print(
                    f"   🧪 Synthetic Data Ratio: {synthetic_impact.get('synthetic_ratio', 0):.2%}"
                )
                estimated_improvement = synthetic_impact.get(
                    "estimated_improvement", {}
                )
                print(
                    f"   📊 Data Diversity Score: {estimated_improvement.get('data_diversity_score', 0):.4f}"
                )
                print(
                    f"   📈 Coverage Enhancement: {estimated_improvement.get('coverage_enhancement', 0):.4f}"
                )
                print(
                    f"   🎯 Quality Boost: {estimated_improvement.get('quality_boost', 0):.4f}"
                )

            # Knowledge injection
            knowledge_injection = instructlab_metrics.get("knowledge_injection", {})
            if knowledge_injection:
                print(
                    f"   🧠 Knowledge Retention: {knowledge_injection.get('knowledge_retention_score', 0):.4f}"
                )
                print(
                    f"   🎯 Domain Accuracy: {knowledge_injection.get('domain_accuracy', 0):.4f}"
                )
                print(
                    f"   ✅ Factual Consistency: {knowledge_injection.get('factual_consistency', 0):.4f}"
                )

            # Overall improvement
            improvement = instructlab_metrics.get("overall_improvement", 0)
            print(f"   🚀 Overall Improvement: {improvement:.2%}")

        else:
            # Standard results display
            print("\n📊 BENCHMARK RESULTS SUMMARY")
            print("=" * 50)
            print(f"🎯 Perplexity: {results.get('perplexity', 'N/A')}")

            if isinstance(results.get("perplexity"), (int, float)):
                print(f"🎯 Perplexity: {results.get('perplexity', 'N/A'):.2f}")
            else:
                print(f"🎯 Perplexity: {results.get('perplexity', 'N/A')}")

            if isinstance(results.get("bleu_4"), (int, float)):
                print(f"🔤 BLEU-4: {results.get('bleu_4', 'N/A'):.2f}")
            else:
                print(f"🔤 BLEU-4: {results.get('bleu_4', 'N/A')}")

            if isinstance(results.get("rouge_l"), (int, float)):
                print(f"🌹 ROUGE-L: {results.get('rouge_l', 'N/A'):.3f}")
            else:
                print(f"🌹 ROUGE-L: {results.get('rouge_l', 'N/A')}")

            if isinstance(results.get("semantic_similarity"), (int, float)):
                print(
                    f"🧠 Semantic Similarity: {results.get('semantic_similarity', 'N/A'):.3f}"
                )
            else:
                print(
                    f"🧠 Semantic Similarity: {results.get('semantic_similarity', 'N/A')}"
                )

            if isinstance(results.get("exact_match"), (int, float)):
                print(f"🎯 Exact Match: {results.get('exact_match', 'N/A'):.3f}")
            else:
                print(f"🎯 Exact Match: {results.get('exact_match', 'N/A')}")

            if isinstance(results.get("avg_inference_time_ms"), (int, float)):
                print(
                    f"⚡ Avg Inference Time: {results.get('avg_inference_time_ms', 'N/A'):.2f} ms"
                )
            else:
                print(
                    f"⚡ Avg Inference Time: {results.get('avg_inference_time_ms', 'N/A')} ms"
                )

        print(f"\n📁 Full results saved to: {results_file}")

        # Show report file if available
        report_files = list(Path(results_dir).glob("benchmark_report_*.txt"))
        if report_files:
            latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
            print(f"📄 Detailed report: {latest_report}")

    except Exception as e:
        print(f"❌ Error reading results: {e}")
        print(f"📁 Results directory: {results_dir}")
        print("=" * 50)

        # Show file locations
        print(f"📄 Detailed Report: {results_dir}/evaluation_report.md")
        print(f"📊 Visualizations: {results_dir}/evaluation_results.png")
        print(f"📋 Predictions: {results_dir}/predictions.json")


def run_enhanced_pipeline(use_instructlab=True, synthetic_samples=100, domain="custom"):
    """Run the enhanced pipeline with InstructLab integration"""
    print("🔬 QLORAX Enhanced Pipeline with InstructLab Integration")
    print("=" * 70)
    print("This enhanced script will:")
    print("1. Validate your setup")
    if use_instructlab:
        print("2. Generate synthetic training data using InstructLab")
        print("3. Combine original and synthetic data")
        print("4. Run enhanced training with mixed dataset")
        print("5. Benchmark with InstructLab-specific metrics")
    else:
        print("2. Run standard training")
        print("3. Benchmark the trained model")
    print("6. Generate comprehensive evaluation report")
    print("=" * 70)

    # Validate setup
    if not validate_setup():
        print("\n❌ Setup validation failed. Please fix the issues above.")
        return 1

    # Check InstructLab availability
    if use_instructlab:
        print("\n🔬 Checking InstructLab integration...")
        try:
            from scripts.instructlab_integration import QLORAXInstructLab

            print("✅ InstructLab integration available")
        except ImportError:
            print("⚠️  InstructLab integration not found. Installing dependencies...")
            if not run_command(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "requirements-instructlab.txt",
                ],
                "Installing InstructLab dependencies",
                check=False,
            ):
                print("❌ Failed to install InstructLab dependencies")
                use_instructlab = False

    # Get user confirmation
    pipeline_type = "enhanced with InstructLab" if use_instructlab else "standard"
    response = input(
        f"\n🤔 Do you want to proceed with {pipeline_type} training? (y/N): "
    )
    if response.lower() not in ["y", "yes"]:
        print("👋 Goodbye!")
        return 0

    # Run training
    if use_instructlab:
        training_success = run_enhanced_training(synthetic_samples, domain)
    else:
        training_success = run_training()

    if not training_success:
        print("\n❌ Training failed. Check the logs above.")
        return 1

    # Check if model was created
    if use_instructlab:
        model_path = f"models/enhanced-qlora/enhanced-qlora-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Check for most recent enhanced model
        enhanced_models = list(Path("models").glob("enhanced-qlora*/enhanced-qlora-*"))
        if enhanced_models:
            model_path = str(max(enhanced_models, key=lambda p: p.stat().st_mtime))
        else:
            model_path = "models/enhanced-qlora"
    else:
        model_path = "models/production-model"

    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return 1

    # Run benchmarking
    if use_instructlab:
        results_dir = run_enhanced_benchmark(model_path)
    else:
        results_dir = run_benchmark(model_path)

    if not results_dir:
        print("\n❌ Benchmarking failed.")
        return 1

    # Show results
    show_results(results_dir, enhanced=use_instructlab)

    print("\n🎉 Enhanced pipeline finished successfully!")
    print("🎯 Your model is trained and benchmarked!")

    if use_instructlab:
        print(f"🧪 Synthetic samples used: {synthetic_samples}")
        print(f"📋 Domain: {domain}")

    print("\n📋 Next steps:")
    print("1. Review the enhanced evaluation report")
    print("2. Deploy your model using scripts/api_server.py or scripts/gradio_app.py")
    print("3. Experiment with different synthetic data parameters")
    print("4. Try knowledge injection with domain-specific documents")

    return 0


def run_enhanced_training(synthetic_samples=100, domain="custom"):
    """Run enhanced training with InstructLab synthetic data generation"""
    print(
        f"\n🧪 Running enhanced training with {synthetic_samples} synthetic samples..."
    )

    # Use enhanced training script
    config_file = "configs/production-config.yaml"
    if not Path(config_file).exists():
        config_file = "configs/default-qlora-config.yml"

    command = [
        sys.executable,
        "scripts/enhanced_training.py",
        "--config",
        config_file,
        "--synthetic-samples",
        str(synthetic_samples),
        "--domain",
        domain,
        "--experiment-name",
        f"enhanced-qlora-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    ]

    return run_command(command, "Enhanced training with InstructLab", check=False)


def run_enhanced_benchmark(model_path):
    """Run enhanced benchmarking with InstructLab metrics"""
    print(f"\n📊 Running enhanced benchmarking on {model_path}...")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/enhanced_benchmark_{timestamp}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Determine test data path
    test_data_paths = ["data/test_data.jsonl", "data/curated.jsonl"]
    test_data = None
    for path in test_data_paths:
        if Path(path).exists():
            test_data = path
            break

    if not test_data:
        print("❌ No test data found")
        return None

    # Run enhanced benchmark
    command = [
        sys.executable,
        "scripts/enhanced_benchmark.py",
        "--model",
        model_path,
        "--test-data",
        test_data,
        "--output",
        results_dir,
        "--instructlab-config",
        "configs/instructlab-config.yaml",
    ]

    if run_command(
        command, "Enhanced benchmarking with InstructLab metrics", check=False
    ):
        return results_dir
    return None


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(
        description="QLORAX Enhanced Quick Start with InstructLab"
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "enhanced"],
        default="enhanced",
        help="Pipeline mode: standard or enhanced with InstructLab",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=100,
        help="Number of synthetic samples to generate (enhanced mode only)",
    )
    parser.add_argument(
        "--domain",
        default="custom",
        help="Domain for synthetic data generation (enhanced mode only)",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run demonstration mode with mock data"
    )

    args = parser.parse_args()

    print("🚀 QLORAX Enhanced Quick Start")
    print("=" * 50)

    if args.demo:
        print("🎭 Running in demonstration mode...")
        return run_demo_mode()

    use_instructlab = args.mode == "enhanced"

    if use_instructlab:
        return run_enhanced_pipeline(
            use_instructlab=True,
            synthetic_samples=args.synthetic_samples,
            domain=args.domain,
        )
    else:
        return run_standard_pipeline()


def run_demo_mode():
    """Run demonstration mode with mock InstructLab integration"""
    print("\n🎭 QLORAX InstructLab Demo Mode")
    print("=" * 50)

    try:
        # Demo InstructLab integration
        print("\n🔬 Demonstrating InstructLab integration...")
        result = run_command(
            [sys.executable, "scripts/instructlab_integration.py"],
            "InstructLab integration demo",
            check=False,
        )

        if result:
            print("✅ InstructLab integration demo completed successfully!")
        else:
            print(
                "⚠️  InstructLab integration demo had issues (this is normal for demo mode)"
            )

        print("\n📋 Demo completed! To run actual training:")
        print("1. Install InstructLab: pip install -r requirements-instructlab.txt")
        print("2. Run: python quick_start.py --mode enhanced")

        return 0

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1


def run_standard_pipeline():
    """Run the standard pipeline without InstructLab"""
    print("🚀 QLORAX Standard Fine-Tuning & Benchmarking Pipeline")
    print("=" * 60)
    print("This script will:")
    print("1. Validate your setup")
    print("2. Run production training")
    print("3. Benchmark the trained model")
    print("4. Generate comprehensive results")
    print("=" * 60)

    # Validate setup
    if not validate_setup():
        print("\n❌ Setup validation failed. Please fix the issues above.")
        return 1

    # Get user confirmation
    response = input(
        "\n🤔 Do you want to proceed with training and benchmarking? (y/N): "
    )
    if response.lower() not in ["y", "yes"]:
        print("👋 Goodbye!")
        return 0

    # Run training
    training_success = run_training()

    if not training_success:
        print("\n❌ Training failed. Check the logs above.")
        return 1

    # Check if model was created
    model_path = "models/production-model"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return 1

    # Run benchmarking
    results_dir = run_benchmark(model_path)

    if not results_dir:
        print("\n❌ Benchmarking failed.")
        return 1

    # Show results
    show_results(results_dir)

    print("\n🎉 Complete pipeline finished successfully!")
    print("🎯 Your model is trained and benchmarked!")
    print("\n📋 Next steps:")
    print("1. Review the evaluation report")
    print("2. Deploy your model using scripts/api_server.py or scripts/gradio_app.py")
    print("3. Try fine-tuning with your own data")
    print("4. Upgrade to enhanced mode: python quick_start.py --mode enhanced")

    return 0


if __name__ == "__main__":
    exit(main())
