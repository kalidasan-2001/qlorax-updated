import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

print("Loading mini dataset...")
# Ensure file exists
if not os.path.exists('data/test/mini_test.jsonl'):
    raise FileNotFoundError("data/test/mini_test.jsonl not found")

dataset = load_dataset('json', data_files='data/test/mini_test.jsonl', split='train')
print(f"Dataset loaded: {len(dataset)} samples")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Tokenizing dataset...")
def tokenize(sample):
    text = f"Instruction: {sample['instruction']}\n\nOutput: {sample['output']}"
    return tokenizer(text, truncation=True, max_length=128, padding='max_length')

dataset = dataset.map(tokenize, remove_columns=['instruction', 'output'])
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print("Loading model (GPT2 for CPU testing)...")
model = AutoModelForCausalLM.from_pretrained("gpt2")

print("Setting up training...")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="./test_output_smoke",
    max_steps=2,
    per_device_train_batch_size=2,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    use_cpu=True  # Force CPU for smoke test compatibility
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=data_collator
)

print("Starting mini training (2 steps)...")
result = trainer.train()
print(f"Training complete! Loss: {result.training_loss:.4f}")
print("Smoke test PASSED - End-to-end training works!")
