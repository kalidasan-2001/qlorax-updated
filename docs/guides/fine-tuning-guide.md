# 🚀 QLORAX Fine-Tuning Guide

## How to Fine-Tune Your Model

### Step 1: Prepare Your Dataset

Your dataset should be in JSONL format (one JSON object per line):

```json
{"input": "Your question or prompt", "output": "The desired response"}
{"input": "Another question", "output": "Another response"}
```

**Current dataset:** `data/my_custom_dataset.jsonl`

### Step 2: Choose Your Training Method

You have 3 options:

#### Option A: Simple Training (Recommended)
```bash
./venv/Scripts/python.exe simple_train.py
```

#### Option B: Batch File (Windows)
```bash
train.bat
```

#### Option C: Axolotl CLI (Advanced)
```bash
./venv/Scripts/python.exe scripts/launch_training.py
```

### Step 3: Monitor Training Progress

During training, you'll see:
- ✅ Model loading progress
- 📊 Training metrics (loss, learning rate)
- 💾 Checkpoint saving
- ⏱️ Training speed and ETA

### Step 4: Training Parameters Explained

**LoRA Configuration:**
- `r=64`: Rank of adaptation (higher = more parameters)
- `alpha=16`: Scaling factor
- `dropout=0.05`: Prevent overfitting
- `target_modules`: Which layers to fine-tune

**Training Settings:**
- `epochs=3`: How many times to go through dataset
- `batch_size=1`: How many examples per batch
- `learning_rate=2e-4`: How fast the model learns
- `warmup_steps=100`: Gradual learning rate increase

### Step 5: After Training Completes

Your fine-tuned model will be saved to:
- `models/tinyllama-qlora-simple/` (Simple training)
- `models/tinyllama-qlora/` (Axolotl training)

### Step 6: Test Your Model

```bash
./venv/Scripts/python.exe scripts/test_model.py
```

### Step 7: Deploy Your Model

**FastAPI Server:**
```bash
./venv/Scripts/python.exe scripts/api_server.py
```

**Gradio Interface:**
```bash
./venv/Scripts/python.exe scripts/gradio_app.py
```

## Customization Options

### Dataset Customization
1. Replace `data/my_custom_dataset.jsonl` with your data
2. Ensure each line has `"input"` and `"output"` fields
3. Use 50-1000+ examples for best results

### Model Customization
Edit these parameters in the training scripts:

**For Better Quality:**
- Increase `r` (rank): 64 → 128
- Increase `epochs`: 3 → 5
- Add more training data

**For Faster Training:**
- Decrease `r` (rank): 64 → 32
- Decrease `max_length`: 512 → 256
- Use fewer `epochs`: 3 → 1

**For Memory Issues:**
- Decrease `batch_size`: 1 → reduce further
- Increase `gradient_accumulation_steps`: 4 → 8
- Use smaller `max_length`: 512 → 256

## Troubleshooting

### Common Issues:

**Out of Memory:**
- Reduce batch size
- Use gradient checkpointing
- Reduce sequence length

**Slow Training:**
- Enable GPU support (install CUDA PyTorch)
- Increase batch size if memory allows
- Use mixed precision training

**Poor Results:**
- Add more training data
- Increase number of epochs
- Adjust learning rate
- Check data quality

### Performance Tips:

1. **More Data = Better Results**: Aim for 100+ examples
2. **Quality over Quantity**: Clean, consistent data works best
3. **Monitor Loss**: Should decrease over time
4. **Save Checkpoints**: Resume training if interrupted

## Current Status

✅ Environment configured
✅ Dependencies installed  
✅ Training scripts ready
✅ Sample dataset provided
✅ Deployment scripts available

**Ready to fine-tune!** 🎯