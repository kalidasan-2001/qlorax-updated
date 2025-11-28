# Reproducible LoRA Training Container
FROM python:3.11-slim

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV CUDA_DETERMINISTIC=1
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command runs reproducible training
CMD ["python", "reproducibility/src/train_lora_reproducible.py"]