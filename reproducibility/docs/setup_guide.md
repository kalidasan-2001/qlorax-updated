# 🛠️ Environment Setup Guide

## Prerequisites

### System Requirements
- **Python**: 3.11+ (recommended: 3.11.7)
- **RAM**: Minimum 8GB, recommended 16GB
- **Storage**: 5GB free space for artifacts
- **Docker**: Latest Docker Desktop (for containerization testing)
- **Git**: For version control and cloning

### Supported Platforms
- ✅ **Windows 10/11** (Primary testing platform)
- ✅ **Linux** (Ubuntu 20.04+, via Docker)
- ✅ **macOS** (Intel/Apple Silicon, via Docker)

## 🖥️ Native Environment Setup

### 1. **Project Setup**

```bash
# Clone repository
git clone https://github.com/kalidasan-2001/qlorax-updated.git
cd qlorax-updated/reproducibility
```

### 2. **Python Environment**

#### **Option A: Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### **Option B: Conda Environment**
```bash
# Create conda environment
conda create -n qlorax-repro python=3.11
conda activate qlorax-repro
```

### 3. **Dependencies Installation**

```bash
# Install core dependencies
pip install -r ../requirements.txt

# Verify critical packages
python -c "import torch, transformers, peft; print('✅ Core packages installed')"

# Verify deterministic capabilities
python -c "import torch; print(f'Deterministic algorithms supported: {torch.are_deterministic_algorithms_enabled()}')" 
```

### 4. **Environment Validation**

```bash
# Test deterministic configuration
python -c "
from src.utils.deterministic_config import DeterministicConfig
DeterministicConfig.setup_deterministic_environment()
valid, messages = DeterministicConfig.validate_deterministic_setup()
print(f'Environment valid: {valid}')
for msg in messages:
    print(f'- {msg}')
"
```

## 🐳 Docker Environment Setup

### 1. **Docker Installation**

#### **Windows**
```powershell
# Download and install Docker Desktop for Windows
# https://docs.docker.com/desktop/install/windows-install/

# Verify installation
docker --version
docker run hello-world
```

#### **Linux (Ubuntu)**
```bash
# Install Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in

# Verify installation
docker --version
docker run hello-world
```

### 2. **Build Docker Image**

```bash
# Navigate to project root
cd qlorax-updated

# Build reproducibility image
docker build -f reproducibility/docker/Dockerfile -t qlorax-reproducibility:latest .

# Verify build
docker images | grep qlorax-reproducibility
```

### 3. **Test Docker Environment**

```bash
# Test basic container execution
docker run --rm qlorax-reproducibility:latest python --version

# Test reproducible training
docker run --rm qlorax-reproducibility:latest

# Test with artifact persistence
mkdir -p ./artifacts
docker run --rm -v "$(pwd)/artifacts:/shared" qlorax-reproducibility:latest bash -c \
  "python reproducibility/src/train_lora_reproducible.py && cp -r artifacts/* /shared/"
```

## 🔧 Configuration & Validation

### 1. **Environment Variables**

Create `.env` file in reproducibility directory:

```bash
# .env file for reproducibility settings
PYTHONHASHSEED=42
CUDA_VISIBLE_DEVICES=""
OMP_NUM_THREADS=1
TORCH_DETERMINISTIC=1
REPRO_MASTER_SEED=42
REPRO_OUTPUT_DIR=./artifacts
```

### 2. **Verification Scripts**

#### **Quick Environment Test**
```bash
# Create test script
cat > test_environment.py << 'EOF'
from src.utils.deterministic_config import DeterministicConfig
import torch
import numpy as np

print("🔍 Testing environment setup...")

# Setup deterministic environment
DeterministicConfig.setup_deterministic_environment(seed=42)

# Validate setup
valid, messages = DeterministicConfig.validate_deterministic_setup()
print(f"Environment valid: {'✅' if valid else '❌'} {valid}")

for msg in messages:
    print(f"  - {msg}")

# Test reproducibility
print("\n🎲 Testing reproducibility...")
torch.manual_seed(42)
np.random.seed(42)

for i in range(3):
    torch.manual_seed(42)
    np.random.seed(42)
    
    torch_val = torch.rand(1).item()
    numpy_val = np.random.rand()
    
    print(f"Run {i+1}: torch={torch_val:.6f}, numpy={numpy_val:.6f}")

print("\n✅ Environment test completed!")
EOF

# Run test
python test_environment.py
```

#### **Full Reproducibility Test**
```bash
# Run multiple training sessions
echo "🔬 Running reproducibility validation..."

for i in {1..3}; do
    echo "Run $i:"
    python src/train_lora_reproducible.py | grep -E "train_loss|eval_loss"
    echo
done
```

### 3. **Troubleshooting Common Issues**

#### **Issue: ImportError for transformers**
```bash
# Solution: Reinstall with specific versions
pip uninstall transformers -y
pip install transformers==4.35.0
```

#### **Issue: Docker permission denied**
```bash
# Linux solution: Add user to docker group
sudo usermod -aG docker $USER
newgrp docker  # or logout/login

# Windows solution: Run Docker Desktop as Administrator
```

#### **Issue: Non-deterministic results**
```bash
# Check for CUDA interference
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Force CPU-only
export CUDA_VISIBLE_DEVICES=""
python src/train_lora_reproducible.py
```

#### **Issue: Missing artifacts directory**
```bash
# Create artifacts directory
mkdir -p reproducibility/artifacts

# Set proper permissions (Linux/Mac)
chmod 755 reproducibility/artifacts
```

## 📋 Validation Checklist

Before running reproducibility experiments, verify:

- [ ] **Python 3.11+** installed and accessible
- [ ] **Virtual environment** activated and configured  
- [ ] **Dependencies** installed via requirements.txt
- [ ] **Docker** installed and running (for container tests)
- [ ] **Artifacts directory** exists and writable
- [ ] **Environment variables** configured correctly
- [ ] **Deterministic test** passes successfully
- [ ] **Quick training run** completes without errors
- [ ] **Docker build** completes successfully
- [ ] **Container execution** runs without issues

## 🚀 Ready to Run

Once setup is complete, you can run the reproducibility study:

```bash
# Single native run
python src/train_lora_reproducible.py

# Multiple validation runs
for i in {1..4}; do
    echo "Validation run $i"
    python src/train_lora_reproducible.py
done

# Docker validation
docker run --rm qlorax-reproducibility:latest

# Comprehensive analysis
python src/analyze_reproducibility.py
```

**🎉 Your environment is now ready for reproducible LoRA fine-tuning experiments!**