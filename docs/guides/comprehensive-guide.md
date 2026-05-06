# 🚀 QLORAX Enhanced Application - Complete Walkthrough Guide

## 📋 **Different Stages of Running the App**

This guide walks you through each stage of the QLORAX enhanced application with InstructLab integration, from initial setup to production deployment.

## Table of Contents
1. [Stage 0: Pre-Flight Setup](#stage-0-pre-flight-setup)
2. [Stage 1: Data Generation Phase](#stage-1-data-generation-phase)
3. [Stage 2: Model Training Phase](#stage-2-model-training-phase)
4. [Stage 3: Model Evaluation Phase](#stage-3-model-evaluation-phase)
5. [Stage 4: Deployment Phase](#stage-4-deployment-phase)
6. [Stage 5: Continuous Operations](#stage-5-continuous-operations)
7. [Stage 6: Advanced Workflows](#stage-6-advanced-workflows)
8. [Stage 7: Troubleshooting & Debug](#stage-7-troubleshooting--debug)

---

## **Stage 0: Pre-Flight Setup** ✈️

### Initial Environment Validation

**Command:**
```powershell
python scripts/validate_system.py
```

**Expected Output:**
```
[SUCCESS] Environment Check Complete
✓ Python 3.13 detected
✓ CUDA available: False (CPU training mode)
✓ All required packages installed
✓ System ready for QLORAX operations
```

### Install Enhanced Dependencies

**Command:**
```powershell
pip install -r requirements-instructlab.txt
```

**Expected Progress:**
- PyTorch 2.9.0 installation
- Transformers 4.57.1 setup
- Gradio 5.49.1 configuration
- Evaluation packages (rouge-score, bert-score)

### Verify Integration

**Command:**
```powershell
python test_integration.py
```

**Expected Results:**
```
[INFO] Testing QLORAX-InstructLab Integration...
✓ InstructLab integration initialized successfully
✓ Synthetic data generation: Working (mock mode)
✓ Training pipeline: Ready
✓ Evaluation suite: Operational
✓ Web interface: Available
✓ API framework: Configured
[SUCCESS] All systems operational!
```

## Data Preparation

### 1. Dataset Format

**JSONL Format (Recommended)**
```json
{"input": "What is machine learning?", "output": "Machine learning is a subset of AI..."}
{"input": "Explain neural networks", "output": "Neural networks are computing systems..."}
```

**CSV Format**
```csv
input,output
"What is machine learning?","Machine learning is a subset of AI..."
"Explain neural networks","Neural networks are computing systems..."
```

**Conversation Format**
```json
{"conversations": [
  {"from": "human", "value": "What is machine learning?"},
  {"from": "assistant", "value": "Machine learning is a subset of AI..."}
]}
```

### 2. Data Quality Guidelines

**Size Recommendations:**
- Minimum: 100 examples
- Good: 1,000+ examples
- Excellent: 10,000+ examples

**Quality Checklist:**
- ✅ Clean, consistent formatting
- ✅ Diverse input patterns
- ✅ High-quality outputs
- ✅ Balanced topic distribution
- ✅ No duplicate examples
- ✅ Appropriate length (50-2048 tokens)

### 3. Data Preprocessing
```python
# Use the built-in data validation
python scripts/validate_dataset.py data/your_dataset.jsonl
```

## Training Configuration

### 1. Model Selection

**Small Models (Fast Training)**
- `microsoft/DialoGPT-small` (117M params)
- `gpt2` (124M params)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B params)

**Medium Models (Balanced)**
- `microsoft/DialoGPT-medium` (354M params)
- `EleutherAI/gpt-neo-1.3B` (1.3B params)

**Large Models (Best Quality)**
- `meta-llama/Llama-2-7b-chat-hf` (7B params)
- `mistralai/Mistral-7B-Instruct-v0.1` (7B params)

### 2. QLoRA Configuration

**Memory-Optimized (4GB RAM)**
```yaml
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
max_length: 512
```

**Balanced (8GB RAM)**
```yaml
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
max_length: 1024
```

**High-Performance (16GB+ RAM)**
```yaml
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
max_length: 2048
```

### 3. Training Hyperparameters

**Conservative (Stable)**
```yaml
learning_rate: 1e-4
num_epochs: 3
warmup_ratio: 0.1
weight_decay: 0.01
lr_scheduler_type: "cosine"
```

**Aggressive (Fast Convergence)**
```yaml
learning_rate: 5e-4
num_epochs: 5
warmup_ratio: 0.05
weight_decay: 0.001
lr_scheduler_type: "linear"
```

## Training Process

### 1. Start Training

**Method A: Production Script (Recommended)**
```bash
python scripts/train_production.py \
  --config configs/production-config.yaml \
  --data data/your_dataset.jsonl \
  --output models/your-model
```

**Method B: Axolotl (Advanced)**
```bash
python -m axolotl.cli.train configs/your-config.yml
```

**Method C: Custom Script**
```bash
python scripts/custom_train.py
```

### 2. Monitor Training

**Weights & Biases (Recommended)**
- Visit: https://wandb.ai/your-username/qlorax-finetuning
- Real-time loss tracking
- System metrics monitoring
- Hyperparameter comparison

**TensorBoard**
```bash
tensorboard --logdir models/your-model/logs
```

**Local Logs**
```bash
tail -f models/your-model/training.log
```

### 3. Key Metrics to Watch

**Training Metrics:**
- Loss: Should decrease steadily
- Learning Rate: Follow schedule
- Gradient Norm: Stay stable (1-10)
- Memory Usage: <90% of available

**Validation Metrics:**
- Perplexity: Lower is better
- BLEU Score: Higher is better (0-100)
- Rouge Score: Higher is better (0-1)

## Evaluation & Benchmarking

### 1. Automatic Evaluation

**Run Full Benchmark**
```bash
python scripts/benchmark.py \
  --model models/your-model \
  --test-data data/test_set.jsonl \
  --output results/benchmark_results.json
```

**Quick Evaluation**
```bash
python scripts/quick_eval.py models/your-model
```

### 2. Evaluation Metrics

**Perplexity** (Language Modeling Quality)
- Range: 1 to ∞ (lower is better)
- Good: <10
- Excellent: <5

**BLEU Score** (Translation/Generation Quality)
- Range: 0-100 (higher is better)
- Good: >20
- Excellent: >40

**ROUGE Score** (Summarization Quality)
- Range: 0-1 (higher is better)
- Good: >0.3
- Excellent: >0.5

**Custom Metrics**
- Task-specific accuracy
- Semantic similarity
- Human evaluation scores

### 3. Benchmark Results Format

```json
{
  "model_info": {
    "name": "your-model",
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "lora_r": 32,
    "training_steps": 1000
  },
  "metrics": {
    "perplexity": 4.23,
    "bleu_score": 34.5,
    "rouge_l": 0.42,
    "exact_match": 0.78,
    "semantic_similarity": 0.85
  },
  "performance": {
    "inference_time_ms": 150,
    "throughput_tokens_per_sec": 45,
    "memory_usage_mb": 2048
  }
}
```

## Advanced Techniques

### 1. Multi-Task Fine-Tuning
```yaml
datasets:
  - path: data/qa_dataset.jsonl
    type: question_answering
    weight: 0.4
  - path: data/summarization_dataset.jsonl  
    type: summarization
    weight: 0.3
  - path: data/chat_dataset.jsonl
    type: conversation
    weight: 0.3
```

### 2. Curriculum Learning
```python
# Progressive difficulty training
python scripts/curriculum_train.py \
  --easy-data data/easy_examples.jsonl \
  --medium-data data/medium_examples.jsonl \
  --hard-data data/hard_examples.jsonl
```

### 3. Knowledge Distillation
```yaml
# Distill from larger model
teacher_model: "meta-llama/Llama-2-7b-chat-hf"
student_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
temperature: 3.0
alpha: 0.7
```

### 4. Retrieval-Augmented Fine-Tuning
```python
# Integrate external knowledge
python scripts/rag_finetune.py \
  --knowledge-base data/knowledge_base.json \
  --retriever-model sentence-transformers/all-MiniLM-L6-v2
```

## Training Commands Reference

### Production Training
```bash
# Full production training with all features
python scripts/train_production.py \
  --config configs/production-config.yaml \
  --data data/training_data.jsonl \
  --validation-data data/validation_data.jsonl \
  --output models/production-model \
  --wandb-project qlorax-production \
  --gradient-checkpointing \
  --fp16 \
  --deepspeed configs/deepspeed_config.json
```

### Hyperparameter Search
```bash
# Automated hyperparameter optimization
python scripts/hyperparameter_search.py \
  --data data/training_data.jsonl \
  --search-space configs/search_space.yaml \
  --trials 50 \
  --output results/hyperopt_results
```

### Model Comparison
```bash
# Compare multiple models
python scripts/model_comparison.py \
  --models models/model1,models/model2,models/model3 \
  --test-data data/test_data.jsonl \
  --output results/comparison.html
```

## Best Practices

### 1. Data Best Practices
- **Validate** data quality before training
- **Split** data: 80% train, 10% validation, 10% test
- **Shuffle** examples to avoid ordering bias
- **Balance** categories and lengths
- **Clean** text (remove artifacts, fix encoding)

### 2. Training Best Practices
- **Start small** with a subset of data
- **Monitor** training closely for first 100 steps
- **Save checkpoints** frequently
- **Use early stopping** to prevent overfitting
- **Log everything** for reproducibility

### 3. Evaluation Best Practices
- **Multiple metrics** for comprehensive evaluation
- **Human evaluation** for quality assessment
- **Error analysis** to identify failure modes
- **A/B testing** for production deployment
- **Continuous monitoring** post-deployment

## Troubleshooting

### Common Issues

**Out of Memory**
```bash
# Reduce batch size and increase gradient accumulation
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

**Slow Training**
```bash
# Enable optimizations
fp16: true
gradient_checkpointing: true
dataloader_num_workers: 4
```

**Poor Performance**
```bash
# Increase model capacity
lora_r: 64
lora_alpha: 128
num_epochs: 10
```

**Overfitting**
```bash
# Add regularization
lora_dropout: 0.1
weight_decay: 0.01
early_stopping_patience: 3
```

### Getting Help

1. **Check logs**: `models/your-model/training.log`
2. **Run diagnostics**: `python scripts/diagnose.py`
3. **Validate setup**: `python scripts/validate_setup.py`
4. **Check issues**: GitHub Issues page
5. **Community**: Discord/Slack channels

---

## Quick Commands Summary

```bash
# Setup and validation
python scripts/validate_setup.py
python scripts/validate_dataset.py data/your_data.jsonl

# Training
python scripts/train_production.py --config configs/your-config.yaml

# Evaluation
python scripts/benchmark.py --model models/your-model
python scripts/quick_eval.py models/your-model

# Deployment
python scripts/api_server.py --model models/your-model
python scripts/gradio_app.py --model models/your-model
```

Ready to fine-tune? Start with the production training script!