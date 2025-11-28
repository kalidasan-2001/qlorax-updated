# 🚀 Production Deployment Guide

## Overview

This guide covers deploying the reproducible LoRA fine-tuning framework in production environments, including CI/CD integration, scaling considerations, and monitoring setup.

## 🏗️ Architecture Overview

```
Production Pipeline:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Reproducible   │───▶│   Model Hub     │
│   (GitHub)      │    │   Training      │    │   (HuggingFace) │
└─────────────────┘    │   (Container)   │    └─────────────────┘
                       └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐
                       │   Validation    │
                       │   & Artifacts   │
                       └─────────────────┘
```

## 🔄 GitHub Actions CI/CD

### 1. **Workflow Configuration**

Create `.github/workflows/reproducible-training.yml`:

```yaml
name: 🔬 Reproducible LoRA Training

on:
  push:
    paths:
      - 'data/**'
      - 'reproducibility/**'
  pull_request:
    paths:
      - 'data/**'
      - 'reproducibility/**'
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM UTC
  workflow_dispatch:
    inputs:
      validation_runs:
        description: 'Number of validation runs'
        required: true
        default: '3'
        type: number

jobs:
  validate-reproducibility:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        environment: [native, docker]
        python-version: [3.11]
    
    steps:
      - name: 🔄 Checkout Repository
        uses: actions/checkout@v4
        
      - name: 🐍 Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
          
      - name: 📦 Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          
      - name: 🔧 Configure Environment
        run: |
          echo "PYTHONHASHSEED=42" >> $GITHUB_ENV
          echo "CUDA_VISIBLE_DEVICES=" >> $GITHUB_ENV
          echo "OMP_NUM_THREADS=1" >> $GITHUB_ENV
          
      - name: ✅ Validate Environment
        run: |
          cd reproducibility
          python -c "
          from src.utils.deterministic_config import DeterministicConfig
          DeterministicConfig.setup_deterministic_environment()
          valid, messages = DeterministicConfig.validate_deterministic_setup()
          print(f'Environment validation: {valid}')
          assert valid, f'Environment validation failed: {messages}'
          "
          
      - name: 🔬 Run Reproducibility Tests (Native)
        if: matrix.environment == 'native'
        run: |
          cd reproducibility
          
          # Run multiple validation sessions
          for i in $(seq 1 ${{ github.event.inputs.validation_runs || 3 }}); do
            echo "Validation run $i"
            python src/train_lora_reproducible.py
          done
          
      - name: 🐳 Build Docker Image
        if: matrix.environment == 'docker'
        run: |
          docker build -f reproducibility/docker/Dockerfile -t qlorax-repro:ci .
          
      - name: 🔬 Run Reproducibility Tests (Docker)
        if: matrix.environment == 'docker'
        run: |
          # Run multiple containerized sessions
          for i in $(seq 1 ${{ github.event.inputs.validation_runs || 2 }}); do
            echo "Docker validation run $i"
            docker run --rm qlorax-repro:ci
          done
          
      - name: 📊 Analyze Results
        run: |
          cd reproducibility
          python src/analyze_reproducibility.py
          
      - name: 📋 Generate Report
        run: |
          cd reproducibility
          
          # Extract key metrics from analysis
          python -c "
          import json
          with open('artifacts/reproducibility_analysis_report.json') as f:
              report = json.load(f)
          
          score = report['overall_analysis']['metrics_reproducibility']['reproducibility_score']
          runs = report['analysis_metadata']['total_runs_analyzed']
          
          print(f'Reproducibility Score: {score:.1%}')
          print(f'Total Runs: {runs}')
          
          # Fail if not perfect reproducibility
          assert score == 1.0, f'Reproducibility failed: {score:.1%}'
          "
          
      - name: 📤 Upload Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: reproducibility-artifacts-${{ matrix.environment }}
          path: |
            reproducibility/artifacts/
            !reproducibility/artifacts/**/*.bin
          retention-days: 30
```

### 2. **Environment Secrets**

Configure repository secrets:

```bash
# GitHub Repository Settings > Secrets and variables > Actions
HUGGINGFACE_TOKEN=hf_your_token_here
DOCKER_HUB_USERNAME=your_username
DOCKER_HUB_TOKEN=your_token
```

## 🏭 Production Infrastructure

### 1. **Kubernetes Deployment**

```yaml
# k8s/reproducible-training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: qlorax-reproducible-training
  namespace: ml-training
spec:
  template:
    spec:
      containers:
      - name: reproducible-trainer
        image: your-registry/qlorax-reproducibility:latest
        env:
        - name: PYTHONHASHSEED
          value: "42"
        - name: CUDA_VISIBLE_DEVICES
          value: ""
        - name: OMP_NUM_THREADS
          value: "1"
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
        volumeMounts:
        - name: artifacts-storage
          mountPath: /app/reproducibility/artifacts
      volumes:
      - name: artifacts-storage
        persistentVolumeClaim:
          claimName: training-artifacts-pvc
      restartPolicy: Never
  backoffLimit: 3
```

### 2. **Monitoring & Observability**

#### **Prometheus Metrics**

```python
# reproducibility/src/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics definitions
training_runs_total = Counter(
    'reproducible_training_runs_total',
    'Total number of reproducible training runs',
    ['environment', 'status']
)

training_duration = Histogram(
    'reproducible_training_duration_seconds',
    'Duration of reproducible training runs',
    ['environment']
)

reproducibility_score = Gauge(
    'reproducibility_score',
    'Reproducibility score (0-1)',
    ['environment']
)

def record_training_metrics(environment: str, duration: float, score: float, success: bool):
    """Record training metrics for monitoring."""
    status = 'success' if success else 'failure'
    
    training_runs_total.labels(environment=environment, status=status).inc()
    training_duration.labels(environment=environment).observe(duration)
    reproducibility_score.labels(environment=environment).set(score)

if __name__ == '__main__':
    # Start metrics server
    start_http_server(8000)
    print("Metrics server started on port 8000")
```

## 📊 Scaling Considerations

### 1. **Horizontal Scaling**

```yaml
# k8s/training-cluster.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reproducible-training-workers
spec:
  replicas: 5
  selector:
    matchLabels:
      app: training-worker
  template:
    metadata:
      labels:
        app: training-worker
    spec:
      containers:
      - name: worker
        image: qlorax-reproducibility:latest
        env:
        - name: WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

### 2. **Resource Management**

```python
# reproducibility/src/resource_manager.py
import psutil
import threading
from typing import Dict, Any

class ResourceManager:
    """Manage computing resources for reproducible training."""
    
    def __init__(self):
        self.max_memory_gb = 8
        self.max_cpu_percent = 80
        
    def check_resources(self) -> Dict[str, Any]:
        """Check available system resources."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'cpu_percent': cpu_percent,
            'ready_for_training': (
                memory.available / (1024**3) >= self.max_memory_gb and
                cpu_percent <= self.max_cpu_percent
            )
        }
    
    def wait_for_resources(self, timeout_minutes: int = 30):
        """Wait for sufficient resources to be available."""
        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            if self.check_resources()['ready_for_training']:
                return True
            time.sleep(10)
        
        return False
```

## 🔐 Security & Compliance

### 1. **Secrets Management**

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: training-secrets
  namespace: ml-training
type: Opaque
data:
  huggingface-token: <base64-encoded-token>
  model-encryption-key: <base64-encoded-key>
```

### 2. **Audit Logging**

```python
# reproducibility/src/audit_logger.py
import logging
import json
from datetime import datetime
from typing import Dict, Any

class AuditLogger:
    """Audit logger for reproducible training runs."""
    
    def __init__(self, log_file: str = "audit.log"):
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_training_start(self, config: Dict[str, Any]):
        """Log training session start."""
        event = {
            'event_type': 'training_start',
            'timestamp': datetime.utcnow().isoformat(),
            'config': config,
            'user': os.environ.get('USER', 'unknown'),
            'environment': os.environ.get('ENVIRONMENT', 'unknown')
        }
        self.logger.info(json.dumps(event))
    
    def log_training_complete(self, results: Dict[str, Any], artifacts: list):
        """Log training session completion."""
        event = {
            'event_type': 'training_complete',
            'timestamp': datetime.utcnow().isoformat(),
            'results': results,
            'artifacts': artifacts
        }
        self.logger.info(json.dumps(event))
```

## 📈 Performance Optimization

### 1. **Caching Strategy**

```python
# reproducibility/src/cache_manager.py
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional

class CacheManager:
    """Manage caching for reproducible training."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, config: dict) -> str:
        """Generate cache key from configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def get_cached_result(self, config: dict) -> Optional[Any]:
        """Retrieve cached training result."""
        cache_key = self._get_cache_key(config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def cache_result(self, config: dict, result: Any):
        """Cache training result."""
        cache_key = self._get_cache_key(config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
```

## 🚦 Quality Gates

### 1. **Automated Quality Checks**

```python
# reproducibility/src/quality_gates.py
from typing import Dict, List, Tuple
import json

class QualityGates:
    """Quality gates for reproducible training."""
    
    def __init__(self):
        self.gates = {
            'reproducibility_score': {'min': 1.0, 'weight': 1.0},
            'training_time': {'max': 300, 'weight': 0.5},  # 5 minutes
            'memory_usage': {'max': 8192, 'weight': 0.3},  # 8GB
            'artifact_count': {'min': 6, 'weight': 0.2}
        }
    
    def evaluate_run(self, results: Dict) -> Tuple[bool, List[str]]:
        """Evaluate training run against quality gates."""
        issues = []
        passed = True
        
        # Check reproducibility score
        repro_score = results.get('reproducibility_score', 0)
        if repro_score < self.gates['reproducibility_score']['min']:
            issues.append(f"Reproducibility score {repro_score} below threshold {self.gates['reproducibility_score']['min']}")
            passed = False
        
        # Check training time
        duration = results.get('training_duration', float('inf'))
        if duration > self.gates['training_time']['max']:
            issues.append(f"Training duration {duration}s exceeds limit {self.gates['training_time']['max']}s")
        
        # Check memory usage
        memory = results.get('peak_memory_mb', 0)
        if memory > self.gates['memory_usage']['max']:
            issues.append(f"Memory usage {memory}MB exceeds limit {self.gates['memory_usage']['max']}MB")
        
        # Check artifact count
        artifact_count = results.get('artifact_count', 0)
        if artifact_count < self.gates['artifact_count']['min']:
            issues.append(f"Artifact count {artifact_count} below minimum {self.gates['artifact_count']['min']}")
        
        return passed, issues
```

## 📞 Support & Maintenance

### 1. **Health Checks**

```python
# reproducibility/src/health_check.py
from flask import Flask, jsonify
import subprocess
import sys

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        # Test environment setup
        result = subprocess.run([
            sys.executable, '-c',
            'from src.utils.deterministic_config import DeterministicConfig; '
            'DeterministicConfig.setup_deterministic_environment(); '
            'print("OK")'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            }), 200
        else:
            return jsonify({
                'status': 'unhealthy',
                'error': result.stderr,
                'timestamp': datetime.utcnow().isoformat()
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### 2. **Automated Recovery**

```bash
#!/bin/bash
# scripts/auto_recovery.sh

set -euo pipefail

echo "🔧 Starting automated recovery..."

# Check Docker service
if ! docker info >/dev/null 2>&1; then
    echo "🐳 Restarting Docker service..."
    sudo systemctl restart docker
    sleep 10
fi

# Rebuild Docker image if corrupted
if ! docker run --rm qlorax-reproducibility:latest python --version >/dev/null 2>&1; then
    echo "🔄 Rebuilding Docker image..."
    docker build -f reproducibility/docker/Dockerfile -t qlorax-reproducibility:latest .
fi

# Clean up disk space
echo "🧹 Cleaning up artifacts..."
find reproducibility/artifacts -name "*.bin" -mtime +7 -delete
docker system prune -f

echo "✅ Recovery completed"
```

---

**🎯 This deployment guide provides a comprehensive framework for moving the reproducible LoRA fine-tuning system from research to production, ensuring scalability, reliability, and maintainability.**