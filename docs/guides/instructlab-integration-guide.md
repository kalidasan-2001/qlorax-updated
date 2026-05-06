# 🔬 QLORAX InstructLab Integration Guide

## Overview

This guide covers the integration of InstructLab with QLORAX, providing synthetic data generation and knowledge injection capabilities to enhance your QLoRA fine-tuning workflows.

## 🎯 What is InstructLab?

InstructLab is a methodology and framework for iterative large language model improvement through:
- **Taxonomy-driven data generation**: Structured approach to creating training data
- **Synthetic data generation**: Automated creation of high-quality training examples
- **Knowledge injection**: Incorporating domain-specific knowledge into models
- **LAB methodology**: Large-scale Alignment for chatBots through iterative improvement

## 🚀 Quick Start

### 1. Installation

```bash
# Install InstructLab dependencies
pip install -r requirements-instructlab.txt

# Verify installation
python scripts/instructlab_integration.py
```

### 2. Basic Usage

```bash
# Run enhanced training with InstructLab
python scripts/quick_start.py --mode enhanced --synthetic-samples 100

# Run with custom domain
python scripts/quick_start.py --mode enhanced --domain machine_learning --synthetic-samples 200

# Demo mode (no actual training)
python scripts/quick_start.py --demo
```

## 📋 Core Components

### 1. InstructLab Integration Module

**File**: `scripts/instructlab_integration.py`

The core integration class `QLORAXInstructLab` provides:

```python
from scripts.instructlab_integration import QLORAXInstructLab

# Initialize integration
instructlab = QLORAXInstructLab("configs/instructlab-config.yaml")

# Create taxonomy for your domain
taxonomy_file = instructlab.create_taxonomy_structure("my_domain")

# Generate synthetic data
synthetic_data = instructlab.generate_synthetic_data(
    taxonomy_path=str(taxonomy_file),
    num_samples=100
)

# Combine with existing data
combined_data = instructlab.integrate_with_qlorax_training(
    synthetic_data_path=str(synthetic_data),
    existing_data_path="data/curated.jsonl"
)
```

### 2. Enhanced Training Pipeline

**File**: `scripts/enhanced_training.py`

Enhanced training with InstructLab support:

```bash
# Enhanced training with synthetic data
python scripts/enhanced_training.py \
    --config configs/production-config.yaml \
    --synthetic-samples 150 \
    --domain technical \
    --experiment-name my-enhanced-model

# With knowledge sources
python scripts/enhanced_training.py \
    --config configs/production-config.yaml \
    --synthetic-samples 100 \
    --domain machine_learning \
    --knowledge-sources docs/ml_guide.md README.md
```

### 3. Enhanced Benchmarking

**File**: `scripts/enhanced_benchmark.py`

Comprehensive evaluation with InstructLab metrics:

```bash
# Enhanced benchmarking
python scripts/enhanced_benchmark.py \
    --model models/enhanced-qlora/my-model \
    --test-data data/test_data.jsonl \
    --output results/enhanced_eval \
    --instructlab-config configs/instructlab-config.yaml
```

## ⚙️ Configuration

### InstructLab Configuration

**File**: `configs/instructlab-config.yaml`

Key configuration sections:

```yaml
# Data generation settings
data_generation:
  model_name: "microsoft/DialoGPT-medium"
  num_samples: 100
  batch_size: 10
  max_length: 512
  temperature: 0.7

# Taxonomy configuration
taxonomy:
  base_path: "instructlab/taxonomy"
  domains:
    - general
    - technical
    - machine_learning

# Training integration
training:
  merge_with_existing: true
  existing_data_weight: 0.7
  synthetic_data_weight: 0.3

# Output paths
output:
  data_dir: "data/instructlab_generated"
  combined_data_file: "data/qlorax_instructlab_combined.jsonl"
```

## 🧪 Synthetic Data Generation

### Creating Taxonomies

Taxonomies define the structure and content areas for synthetic data generation:

```python
# Create a custom taxonomy
taxonomy_data = {
    "version": 1,
    "domain": "machine_learning",
    "created_by": "QLORAX",
    "seed_examples": [
        {
            "question": "What is supervised learning?",
            "answer": "Supervised learning is a machine learning approach..."
        }
    ],
    "knowledge_areas": [
        "supervised_learning",
        "unsupervised_learning", 
        "deep_learning"
    ]
}
```

### Domain-Specific Generation

Generate data for specific domains:

```bash
# Machine learning domain
python scripts/enhanced_training.py \
    --domain machine_learning \
    --synthetic-samples 200

# Software engineering domain  
python scripts/enhanced_training.py \
    --domain software_engineering \
    --synthetic-samples 150
```

## 🧠 Knowledge Injection

### Using Knowledge Sources

Inject domain-specific knowledge from documents:

```python
# Create knowledge taxonomy from documents
knowledge_taxonomy = instructlab.create_knowledge_taxonomy(
    domain="technical_docs",
    knowledge_docs=[
        "docs/api_reference.md",
        "docs/user_guide.md",
        "README.md"
    ]
)
```

### Knowledge-Enhanced Training

```bash
# Training with knowledge injection
python scripts/enhanced_training.py \
    --config configs/production-config.yaml \
    --knowledge-sources docs/domain_guide.md docs/examples.md \
    --domain custom_domain \
    --synthetic-samples 100
```

## 📊 Evaluation and Metrics

### InstructLab-Specific Metrics

The enhanced benchmarking provides additional metrics:

1. **Synthetic Data Impact**
   - Data diversity score
   - Coverage enhancement
   - Quality improvement

2. **Knowledge Injection Effectiveness**
   - Knowledge retention score
   - Domain accuracy
   - Factual consistency

3. **Overall Improvement**
   - Compared to baseline models
   - Iterative improvement tracking

### Interpreting Results

Example enhanced benchmark output:

```
🔬 InstructLab Enhancement Metrics:
   🧪 Synthetic Data Ratio: 30.00%
   📊 Data Diversity Score: 0.8500
   📈 Coverage Enhancement: 0.2300
   🧠 Knowledge Retention: 0.7800
   🎯 Domain Accuracy: 0.8200
   🚀 Overall Improvement: 15.00%
```

## 🔧 Advanced Usage

### Custom Data Generators

Create custom synthetic data generators:

```python
class CustomDataGenerator:
    def __init__(self, domain):
        self.domain = domain
    
    def generate_samples(self, num_samples):
        # Custom generation logic
        pass

# Register custom generator
instructlab.register_generator("custom", CustomDataGenerator)
```

### Iterative Improvement

Implement iterative model improvement:

```bash
# Stage 1: Initial training
python scripts/enhanced_training.py \
    --config configs/stage1-config.yaml \
    --synthetic-samples 100

# Stage 2: Improved training with feedback
python scripts/enhanced_training.py \
    --config configs/stage2-config.yaml \
    --synthetic-samples 150 \
    --baseline-model models/stage1-model
```

### Distributed Generation

For large-scale synthetic data generation:

```yaml
# instructlab-config.yaml
advanced:
  distributed:
    enabled: true
    num_workers: 4
    backend: "multiprocessing"
```

## 🛠️ Troubleshooting

### Common Issues

1. **InstructLab Not Found**
   ```bash
   # Install dependencies
   pip install -r requirements-instructlab.txt
   
   # Verify installation
   python -c "import instructlab; print('OK')"
   ```

2. **Taxonomy Creation Fails**
   - Check directory permissions
   - Verify YAML syntax
   - Ensure required fields are present

3. **Synthetic Data Generation Issues**
   - Check model availability
   - Verify taxonomy structure
   - Monitor memory usage

4. **Training Integration Problems**
   - Verify data format compatibility
   - Check file paths
   - Review configuration settings

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
instructlab = QLORAXInstructLab("configs/instructlab-config.yaml")
```

### Performance Optimization

Optimize generation performance:

```yaml
# instructlab-config.yaml
data_generation:
  batch_size: 20  # Increase batch size
  max_length: 256  # Reduce max length
  
development:
  profile_execution: true  # Enable profiling
```

## 📈 Best Practices

### 1. Data Quality

- **Validate Generated Data**: Always run validation on synthetic data
- **Balance Ratios**: Maintain 70/30 split between original and synthetic data
- **Domain Consistency**: Keep synthetic data aligned with your domain

### 2. Taxonomy Design

- **Start Small**: Begin with simple taxonomies and iterate
- **Use Seed Examples**: Provide high-quality seed examples
- **Domain Focus**: Create domain-specific taxonomies

### 3. Training Strategy

- **Gradual Integration**: Start with small amounts of synthetic data
- **Monitor Performance**: Track metrics throughout training
- **Iterate and Improve**: Use feedback to refine generation

### 4. Evaluation

- **Comprehensive Testing**: Use multiple evaluation metrics
- **Domain-Specific Tests**: Create domain-specific test sets
- **Baseline Comparison**: Always compare against baseline models

## 🔄 Workflow Examples

### Complete Enhancement Workflow

```bash
# 1. Setup
pip install -r requirements-instructlab.txt

# 2. Generate synthetic data
python scripts/instructlab_integration.py

# 3. Enhanced training
python scripts/enhanced_training.py \
    --config configs/production-config.yaml \
    --synthetic-samples 200 \
    --domain machine_learning

# 4. Enhanced evaluation
python scripts/enhanced_benchmark.py \
    --model models/enhanced-qlora/latest \
    --test-data data/test_data.jsonl \
    --output results/enhanced_eval

# 5. Deploy enhanced model
python scripts/api_server.py --model models/enhanced-qlora/latest
```

### Research and Development Workflow

```bash
# Experiment with different synthetic data amounts
for samples in 50 100 150 200; do
    python scripts/enhanced_training.py \
        --synthetic-samples $samples \
        --experiment-name "experiment-$samples-samples"
done

# Compare results
python scripts/compare_experiments.py \
    --experiment-dir models/enhanced-qlora/
```

## 📚 Additional Resources

### InstructLab Documentation
- [InstructLab GitHub](https://github.com/instructlab/instructlab)
- [InstructLab Documentation](https://instructlab.ai/docs/)
- [LAB Methodology Paper](https://arxiv.org/abs/2403.01081)

### QLORAX Integration
- [QLORAX Documentation](README.md)
- [Configuration Guide](COMPREHENSIVE_GUIDE.md)
- [Benchmarking Guide](scripts/benchmark.py)

### Community and Support
- [GitHub Discussions](https://github.com/your-repo/discussions)
- [Issues and Bug Reports](https://github.com/your-repo/issues)

## 🤝 Contributing

To contribute to the InstructLab integration:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-repo/QLORAX.git
cd QLORAX

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-instructlab.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/test_instructlab_integration.py
```

---

**📧 Need Help?** 

Create an issue on GitHub or check our documentation for more detailed examples and troubleshooting guides.