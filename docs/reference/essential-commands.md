# 🚀 QLORAX Enhanced System - Essential Commands Guide

## 🎯 Core Operations

### 1. 🔧 System Setup & Validation
```bash
# Activate virtual environment (Windows)
& "C:/CloudSpace/OneDrive - neokloud/Desktop/arjun.cloud/projects/QLORAX2/venv/Scripts/Activate.ps1"

# Test system integration
python test_integration.py

# Set UTF-8 encoding for Windows
chcp 65001
```

### 2. 📊 Data Generation & Management
```bash
# Generate synthetic data with InstructLab integration
python scripts/instructlab_integration.py --samples 20 --domain "machine_learning"

# Generate for specific domains
python scripts/instructlab_integration.py --samples 15 --domain "artificial_intelligence"
python scripts/instructlab_integration.py --samples 25 --domain "data_science"
python scripts/instructlab_integration.py --samples 30 --domain "deep_learning"

# Check generated synthetic data
dir data\instructlab_generated\

# View synthetic data content
type data\instructlab_generated\synthetic_data_*.jsonl | findstr /C:"instruction"
```

### 3. 🤖 Model Training
```bash
# Run enhanced training pipeline (RECOMMENDED)
python scripts/run_enhanced_training.py --samples 15 --domain "machine_learning"

# Advanced training with more samples
python scripts/run_enhanced_training.py --samples 25 --domain "artificial_intelligence"

# Custom domain training
python scripts/run_enhanced_training.py --samples 20 --domain "your_custom_domain"

# Original enhanced training (if needed)
chcp 65001; python scripts/quick_start.py --mode enhanced --synthetic-samples 10 --domain "machine_learning"
```

### 4. 📈 Model Evaluation & Benchmarking
```bash
# Run comprehensive benchmark
python scripts/enhanced_benchmark.py --model models/enhanced-qlora-demo --test-data data/training_data.jsonl --output results/benchmark_results.json

# Quick evaluation
python scripts/enhanced_benchmark.py --model models/enhanced-qlora-demo --test-data data/training_data.jsonl --output results/

# Compare multiple models
python scripts/enhanced_benchmark.py --model models/demo-qlora-model --test-data data/training_data.jsonl --output results/baseline/
```

### 5. 🌐 Model Deployment & Testing
```bash
# Start API server
python scripts/api_server.py

# Start Gradio web interface
python scripts/gradio_app.py

# Test API endpoints (in another terminal)
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "What is machine learning?"}'
```

### 6. 📊 Results & Monitoring
```bash
# View training summary
type training_summary.json

# Check model metadata
type models\enhanced-qlora-demo\training_metadata.json

# View benchmark results
dir results\
type results\benchmark_results.json\enhanced_benchmark_results_*.json

# Check logs
dir models\enhanced-qlora-demo\
```

## 🎮 Quick Demo Workflows

### 🚀 Full Pipeline Demo (5 minutes)
```bash
# 1. Generate synthetic data
python scripts/instructlab_integration.py --samples 10 --domain "AI"

# 2. Run enhanced training
python scripts/run_enhanced_training.py --samples 10 --domain "AI"

# 3. Benchmark the model
python scripts/enhanced_benchmark.py --model models/enhanced-qlora-demo --test-data data/training_data.jsonl --output results/

# 4. Start API server
python scripts/api_server.py
```

### 🔬 Research & Development Workflow
```bash
# 1. Test integration
python test_integration.py

# 2. Generate domain-specific data
python scripts/instructlab_integration.py --samples 25 --domain "deep_learning"

# 3. Train enhanced model
python scripts/run_enhanced_training.py --samples 25 --domain "deep_learning"

# 4. Comprehensive evaluation
python scripts/enhanced_benchmark.py --model models/enhanced-qlora-demo --test-data data/training_data.jsonl --output results/research/

# 5. Deploy for testing
python scripts/gradio_app.py
```

## 🛠️ Maintenance & Utilities

### 📁 File Management
```bash
# List all models
dir models\

# Check data files
dir data\
dir data\instructlab_generated\

# Clean old results
rmdir /s results\old\

# Backup current model
xcopy models\enhanced-qlora-demo models\backup\enhanced-qlora-demo-backup\ /E /I
```

### 🔍 Debugging & Troubleshooting
```bash
# Check Python environment
python --version
pip list | findstr torch
pip list | findstr transformers

# Test individual components
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Verbose training for debugging
python scripts/run_enhanced_training.py --samples 5 --domain "test" --verbose

# Check system resources
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"
```

## 🎯 Production Workflows

### 🚀 Production Deployment
```bash
# 1. Generate production data
python scripts/instructlab_integration.py --samples 50 --domain "production_domain"

# 2. Train production model
python scripts/run_enhanced_training.py --samples 50 --domain "production_domain"

# 3. Comprehensive evaluation
python scripts/enhanced_benchmark.py --model models/enhanced-qlora-demo --test-data data/training_data.jsonl --output results/production/

# 4. Deploy API server
python scripts/api_server.py --port 8000 --host 0.0.0.0
```

### 📊 Performance Monitoring
```bash
# Monitor training progress
type training_summary.json

# Check model performance
python scripts/enhanced_benchmark.py --model models/enhanced-qlora-demo --test-data data/training_data.jsonl --output results/monitoring/

# API health check
curl http://localhost:8000/health
```

## 🎨 Customization Commands

### 🎛️ Custom Training Configurations
```bash
# Custom domain training
python scripts/run_enhanced_training.py --samples 30 --domain "healthcare"
python scripts/run_enhanced_training.py --samples 20 --domain "finance"
python scripts/run_enhanced_training.py --samples 25 --domain "education"

# Batch processing
for domain in ("AI" "ML" "DL" "NLP"); do python scripts/run_enhanced_training.py --samples 15 --domain $domain; done
```

### 📈 Advanced Evaluation
```bash
# Detailed benchmark with baseline comparison
python scripts/enhanced_benchmark.py --model models/enhanced-qlora-demo --test-data data/training_data.jsonl --output results/detailed/ --save-detailed

# Cross-validation testing
python scripts/enhanced_benchmark.py --model models/enhanced-qlora-demo --test-data data/test_synthetic/synthetic_test.jsonl --output results/cross_val/
```

## 📋 Daily Usage Commands

### ☀️ Morning Routine
```bash
# Check system status
python test_integration.py

# Generate fresh synthetic data
python scripts/instructlab_integration.py --samples 15 --domain "daily_tasks"

# Quick training update
python scripts/run_enhanced_training.py --samples 15 --domain "daily_tasks"
```

### 🌙 Evening Routine
```bash
# Run comprehensive benchmark
python scripts/enhanced_benchmark.py --model models/enhanced-qlora-demo --test-data data/training_data.jsonl --output results/daily/

# Backup progress
xcopy models\enhanced-qlora-demo models\backups\$(date +%Y%m%d)\ /E /I

# Check results
type training_summary.json
```

---

## 🎯 **Most Important Commands (Top 5)**

1. **Full Training Pipeline**: `python scripts/run_enhanced_training.py --samples 15 --domain "machine_learning"`
2. **Synthetic Data Generation**: `python scripts/instructlab_integration.py --samples 20 --domain "AI"`
3. **Model Evaluation**: `python scripts/enhanced_benchmark.py --model models/enhanced-qlora-demo --test-data data/training_data.jsonl --output results/`
4. **API Deployment**: `python scripts/api_server.py`
5. **System Testing**: `python test_integration.py`

**Use these commands to operate your enhanced QLORAX system efficiently! 🚀**