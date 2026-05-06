# QLORAX Enhanced

A thesis-backed **QLoRA fine-tuning suite** for dataset preparation, reproducible training, evaluation gating, and Docker-based deployment.

## Highlights

- **Hybrid dataset pipeline** with curated and synthetic samples
- **Reproducibility artifacts** for fixed-prompt, repeatable training runs
- **Evaluation gate** outputs for quality and regression checks
- **Docker support** for training and serving
- **CI/CD workflows** for automation and verification
- **Documentation** for setup, training, evaluation, and troubleshooting

## Repository layout

| Path | Purpose |
| --- | --- |
| `scripts/` | Training, evaluation, validation, API, and utility scripts |
| `configs/` | Training and runtime configuration files |
| `data/variants/hybrid_70_30.jsonl` | Final hybrid dataset used for thesis experiments |
| `reproducibility/manifests/` | Reproducibility evidence and comparison outputs |
| `reproducibility/configs/` | Reproducibility configuration files |
| `rq4_evaluation/manifests/` | Evaluation-gate results and comparisons |
| `docs/` | Guides, setup notes, references, and troubleshooting |
| `.github/workflows/` | GitHub Actions workflows |
| `figures/summary_metrics.csv` | Latest summary metrics snapshot |
| `Dockerfile.training`, `Dockerfile.serve`, `docker-compose.yml` | Container setup |

## Quick start

### 1) Clone the repository

```bash
git clone https://github.com/kalidasan-2001/qlorax-updated.git
cd qlorax-enhanced
```

### 2) Create and activate a virtual environment

**Windows**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

Core environment:

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -r requirements-train.txt
pip install -r requirements-serve.txt
pip install -r requirements-instructlab.txt
pip install -r requirements-dev.txt
```

### 4) Validate the environment

```bash
python scripts/validate_system.py
```

### 5) Run a quick training flow

```bash
python scripts/quick_start.py
```

Enhanced training:

```bash
python scripts/run_enhanced_training.py --config configs/production-config.yaml
```

## Common tasks

### Training

- `scripts/quick_start.py` — simple entry point
- `scripts/run_enhanced_training.py` — enhanced training workflow
- `scripts/enhanced_training.py` — training orchestrator
- `scripts/train_production.py` — production-oriented training
- `scripts/instructlab_integration.py` — synthetic data and InstructLab integration

### Evaluation

- `scripts/enhanced_benchmark.py` — benchmark and evaluation flow
- `scripts/quality_gates.py` — gate checks and threshold logic
- `scripts/validate_system.py` — environment and dependency validation
- `rq4_evaluation/manifests/` — evaluation evidence

### Serving

- `scripts/api_server.py` — API server
- `scripts/web_demo.py` — web UI demo
- `docker-compose.yml` — container orchestration

## Reproducibility and evidence

This repository includes thesis-critical evidence for:

- dataset construction
- reproducible training execution
- evaluation-gate checks
- CI/CD validation

Key evidence locations:

- `data/variants/hybrid_70_30.jsonl`
- `reproducibility/manifests/`
- `reproducibility/configs/`
- `rq4_evaluation/manifests/`

## Latest summary metrics

See `figures/summary_metrics.csv` for the latest snapshot. Selected values:

- Semantic similarity: `0.7697`
- BLEU-1: `28.5416`
- BLEU-2: `21.6344`
- BLEU-4: `18.7194`
- ROUGE-1: `0.4019`
- ROUGE-2: `0.2893`
- ROUGE-L: `0.3531`
- Exact match: `0.0`
- Eval loss: `1.6927`
- Perplexity: `5.4343`

## Docker

Build the training image:

```bash
docker build -f Dockerfile.training -t qlorax-train .
```

Build the serving image:

```bash
docker build -f Dockerfile.serve -t qlorax-serve .
```

Start the stack:

```bash
docker-compose up -d
```

## Documentation

- `docs/guides/comprehensive-guide.md`
- `docs/guides/fine-tuning-guide.md`
- `docs/guides/instructlab-integration-guide.md`
- `docs/guides/walkthrough-stages.md`
- `docs/setup/installation-complete.md`
- `docs/setup/ci-cd-setup.md`
- `docs/reference/essential-commands.md`
- `docs/reference/instructlab-integration-summary.md`
- `docs/reference/run-complete.md`
- `docs/troubleshooting/error-resolution.md`
- `docs/troubleshooting/training-fix.md`

## Development

```bash
pytest tests/ -v
black .
isort .
mypy scripts/
```

## License

MIT License. See `LICENSE`.
