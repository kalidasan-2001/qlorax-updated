# 🚀 QLORAX: Production QLoRA Fine-Tuning & Benchmarking Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Overview

QLORAX is a comprehensive MLOps project for production-grade fine-tuning of large language models using QLoRA (Quantized Low-Rank Adaptation). This project provides end-to-end infrastructure for data preparation, model training, comprehensive evaluation, and deployment with professional benchmarking capabilities.

### ✨ Key Features

- **🎯 Production Training**: Memory-efficient QLoRA with comprehensive logging and monitoring
- **📊 Advanced Benchmarking**: BLEU, ROUGE, perplexity, semantic similarity, and performance metrics
- **🔧 Multiple Deployment Options**: FastAPI REST API, Gradio web interface, and Jupyter notebooks
- **📈 Real-time Monitoring**: Weights & Biases integration with experiment tracking
- **🛡️ Robust Pipeline**: Error handling, checkpointing, early stopping, and resume functionality
- **📚 Comprehensive Documentation**: Step-by-step guides, best practices, and troubleshooting

## ✅ System Status

**🎉 All errors resolved! System is fully operational.**

Recent fixes applied:
- ✅ Fixed Weights & Biases configuration issues
- ✅ Corrected data tokenization for causal language modeling  
- ✅ Updated API compatibility with current transformers
- ✅ Validated complete training pipeline works correctly

📋 See `ERROR_RESOLUTION.md` for detailed technical information.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ (3.13 recommended)
- 8GB+ RAM (16GB+ recommended for larger models)
- CUDA GPU (optional but recommended for faster training)

### 1. Environment Setup
```bash
# Activate virtual environment (already created)
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Linux/Mac

# Install additional dependencies
python scripts/setup_dependencies.py

# Validate setup
python scripts/validate_setup.py
```

### 2. Complete Pipeline (Recommended)
```bash
# Run full training + benchmarking pipeline
python scripts/quick_start.py
```
This will:
- ✅ Validate your environment
- 🎯 Train a QLoRA model on sample data
- 📊 Run comprehensive benchmarking
- 📝 Generate detailed evaluation reports
- 📈 Create visualization plots

### 3. Manual Training & Benchmarking
```bash
# Production training with full monitoring
python scripts/train_production.py \
  --config configs/production-config.yaml \
  --wandb-project my-project

# Comprehensive benchmarking
python scripts/benchmark.py \
  --model models/production-model \
  --test-data data/test_data.jsonl \
  --output results/my_benchmark
```

## 📁 Project Structure

```
QLORAX/
├── 📚 Documentation
│   ├── COMPREHENSIVE_GUIDE.md      # Complete fine-tuning guide
│   ├── FINE_TUNING_GUIDE.md       # Step-by-step tutorial
│   └── README.md                   # This file
│
├── 🎯 Training & Evaluation
│   ├── scripts/
│   │   ├── train_production.py     # Production training script
│   │   ├── benchmark.py            # Comprehensive benchmarking
│   │   ├── test_model.py          # Model testing utilities
│   │   └── validate_setup.py      # Environment validation
│   │
│   ├── configs/
│   │   ├── production-config.yaml  # Production training config
│   │   └── default-qlora-config.yml # Axolotl configuration
│   │
│   └── data/
│       ├── training_data.jsonl     # Training dataset
│       ├── test_data.jsonl        # Test dataset
│       └── my_custom_dataset.jsonl # Your custom data
│
├── 🚀 Deployment
│   ├── scripts/
│   │   ├── api_server.py          # FastAPI REST API
│   │   ├── gradio_app.py          # Web interface
│   │   └── jupyter notebooks/      # Interactive development
│   │
├── 📊 Models & Results
│   ├── models/                     # Trained models
│   ├── results/                    # Benchmark results
│   └── logs/                      # Training logs
│
└── 🛠️ Utilities
    ├── quick_start.py             # Complete pipeline
    ├── setup_dependencies.py     # Install additional deps
    └── tutorial.py               # Interactive tutorial
```

## 📊 Benchmarking Capabilities

QLORAX provides comprehensive model evaluation across multiple dimensions:

### 🎯 Language Modeling Quality
- **Perplexity**: Measures model's uncertainty (lower is better)
- **Cross-entropy Loss**: Training convergence metric

### 🔤 Generation Quality  
- **BLEU Scores**: N-gram overlap with references (BLEU-1, 2, 4)
- **ROUGE Scores**: Recall-oriented metrics (ROUGE-1, 2, L)
- **Exact Match**: Percentage of perfect predictions

### 🧠 Semantic Quality
- **Semantic Similarity**: Cosine similarity of sentence embeddings
- **Contextual Understanding**: Domain-specific evaluation

### ⚡ Performance Metrics
- **Inference Speed**: Average time per prediction
- **Throughput**: Samples processed per second  
- **Memory Usage**: Peak memory consumption
- **Model Size**: Storage requirements

### 📈 Visualization & Reporting
- **Interactive Plots**: Metrics visualization with matplotlib/seaborn
- **Comprehensive Reports**: Markdown reports with interpretation
- **Weights & Biases**: Real-time experiment tracking
- **Comparative Analysis**: Multi-model comparison

## 🎯 Training Configurations

### Memory-Optimized (4GB RAM)
```yaml
lora_r: 16
lora_alpha: 32
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
max_length: 512
```

### Balanced (8GB RAM)  
```yaml
lora_r: 32
lora_alpha: 64
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
max_length: 1024
```

### High-Performance (16GB+ RAM)
```yaml
lora_r: 64
lora_alpha: 128
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
max_length: 2048
```

## 🚀 Deployment Options

### 1. FastAPI REST API
```bash
python scripts/api_server.py --model models/your-model
# Access at: http://localhost:8000/docs
```

### 2. Gradio Web Interface
```bash
python scripts/gradio_app.py --model models/your-model
# Access at: http://localhost:7860
```

### 3. Jupyter Notebook
```bash
jupyter lab notebooks/
# Interactive development and testing
```

## 📈 Sample Benchmark Results

```json
{
  "model_info": {
    "name": "tinyllama-qlora-v1",
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "lora_r": 32,
    "training_steps": 500
  },
  "metrics": {
    "perplexity": 4.23,
    "bleu_4": 34.5,
    "rouge_l": 0.42,
    "semantic_similarity": 0.85,
    "exact_match": 0.78
  },
  "performance": {
    "avg_inference_time_ms": 150,
    "throughput_samples_per_sec": 6.7,
    "memory_usage_mb": 2048
  }
}
```

## 🛠️ Custom Data Preparation

### 1. Data Format (JSONL)
```json
{"input": "What is machine learning?", "output": "Machine learning is..."}
{"input": "Explain neural networks", "output": "Neural networks are..."}
```

### 2. Data Quality Guidelines
- **Size**: 100+ examples (1000+ recommended)
- **Quality**: Clean, consistent, diverse examples
- **Length**: 50-2048 tokens per example
- **Balance**: Even distribution across topics/difficulty

### 3. Validation
```bash
python scripts/validate_dataset.py data/your_dataset.jsonl
```

## 📚 Documentation

- **[Comprehensive Guide](COMPREHENSIVE_GUIDE.md)**: Complete fine-tuning documentation
- **[Tutorial](tutorial.py)**: Interactive step-by-step guide
- **[Best Practices](COMPREHENSIVE_GUIDE.md#best-practices)**: Training and evaluation tips
- **[Troubleshooting](COMPREHENSIVE_GUIDE.md#troubleshooting)**: Common issues and solutions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black scripts/ --line-length 88
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - QLoRA training framework
- [Hugging Face](https://huggingface.co/) - Transformers and model hub
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [Weights & Biases](https://wandb.ai/) - Experiment tracking

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/qlorax/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/qlorax/discussions)
- **Documentation**: [Comprehensive Guide](COMPREHENSIVE_GUIDE.md)

---

**Ready to fine-tune?** Start with `python scripts/quick_start.py` and follow the interactive guide! 🚀