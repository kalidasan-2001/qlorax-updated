#!/usr/bin/env python3
"""Update enhanced_training.py with embedded evaluation"""

import sys
from pathlib import Path

# Read the original file
file_path = Path("scripts/enhanced_training.py")
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    lines = f.readlines()

# Reset file pointer and read lines
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Backup
backup_path = file_path.with_suffix('.py.backup')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"✓ Backup created: {backup_path}")

# Find where to insert evaluation functions (before EnhancedQLORAXTrainer class)
insert_pos = None
for i, line in enumerate(lines):
    if "class EnhancedQLORAXTrainer:" in line:
        insert_pos = i
        break

if not insert_pos:
    print("Error: Could not find EnhancedQLORAXTrainer class")
    sys.exit(1)

# Evaluation functions to insert
eval_functions = '''
# ============================================================================
# EMBEDDED EVALUATION FUNCTIONS (for comprehensive model evaluation)
# ============================================================================

def compute_perplexity_eval(model, tokenizer, dataset, max_length=512):
    """Compute perplexity on evaluation dataset"""
    import torch
    import numpy as np
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for example in dataset:
            text = f"### Instruction:\\n{example['instruction']}\\n\\n### Output:\\n{example['output']}\\n"
            inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                             max_length=max_length).to(model.device)
            outputs = model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
            total_tokens += inputs['input_ids'].size(1)
    
    avg_loss = total_loss / total_tokens
    return np.exp(avg_loss)


def generate_predictions_eval(model, tokenizer, dataset, max_new_tokens=128):
    """Generate predictions for evaluation"""
    import torch
    
    model.eval()
    predictions = []
    
    for example in dataset:
        instruction = example['instruction']
        prompt = f"### Instruction:\\n{instruction}\\n\\n### Output:\\n"
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = generated_text.split("### Output:")[-1].strip()
        predictions.append(output)
    
    return predictions


def compute_generation_metrics(predictions, references):
    """Compute BLEU, ROUGE, exact match, semantic similarity"""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    metrics = {}
    
    # BLEU scores
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import word_tokenize
        
        smoothing = SmoothingFunction().method1
        bleu_1_scores, bleu_2_scores, bleu_4_scores = [], [], []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = [word_tokenize(ref.lower())]
            
            bleu_1_scores.append(sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing))
            bleu_2_scores.append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing))
            bleu_4_scores.append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing))
        
        metrics['bleu_1'] = np.mean(bleu_1_scores) * 100
        metrics['bleu_2'] = np.mean(bleu_2_scores) * 100
        metrics['bleu_4'] = np.mean(bleu_4_scores) * 100
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        metrics['bleu_1'] = metrics['bleu_2'] = metrics['bleu_4'] = 0.0
    
    # ROUGE scores
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge_1_scores, rouge_2_scores, rouge_l_scores = [], [], []
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge_1_scores.append(scores['rouge1'].fmeasure)
            rouge_2_scores.append(scores['rouge2'].fmeasure)
            rouge_l_scores.append(scores['rougeL'].fmeasure)
        
        metrics['rouge_1'] = np.mean(rouge_1_scores)
        metrics['rouge_2'] = np.mean(rouge_2_scores)
        metrics['rouge_l'] = np.mean(rouge_l_scores)
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        metrics['rouge_1'] = metrics['rouge_2'] = metrics['rouge_l'] = 0.0
    
    # Exact match
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    metrics['exact_match'] = matches / len(predictions)
    
    # Semantic similarity
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        pred_emb = model.encode(predictions, show_progress_bar=False)
        ref_emb = model.encode(references, show_progress_bar=False)
        similarities = [cosine_similarity([p], [r])[0][0] for p, r in zip(pred_emb, ref_emb)]
        metrics['semantic_similarity'] = np.mean(similarities)
    except Exception as e:
        logger.warning(f"Semantic similarity failed: {e}")
        metrics['semantic_similarity'] = 0.0
    
    return metrics


def run_comprehensive_evaluation(model, tokenizer, eval_dataset, training_metrics, output_dir):
    """Run comprehensive evaluation and save results"""
    import json
    import time
    import torch
    from pathlib import Path
    
    logger.info("Running comprehensive evaluation...")
    output_path = Path(output_dir)
    metrics_dir = output_path / 'metrics'
    metrics_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate predictions
    start_time = time.time()
    predictions = generate_predictions_eval(model, tokenizer, eval_dataset)
    inference_time = time.time() - start_time
    references = [ex['output'] for ex in eval_dataset]
    
    # Compute metrics
    perplexity = compute_perplexity_eval(model, tokenizer, eval_dataset)
    gen_metrics = compute_generation_metrics(predictions, references)
    
    # Efficiency metrics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    
    # Compile results
    evaluation_results = {
        'language_modeling': {
            'train_loss': training_metrics.get('train_loss', 0),
            'eval_loss': training_metrics.get('eval_loss', 0),
            'perplexity': float(perplexity),
        },
        'generation_quality': gen_metrics,
        'efficiency': {
            'total_training_time_sec': training_metrics.get('train_runtime', 0),
            'throughput_samples_per_sec': len(eval_dataset) / training_metrics.get('train_runtime', 1),
            'avg_inference_time_ms': (inference_time / len(eval_dataset)) * 1000,
            'peak_gpu_memory_mb': float(gpu_memory),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100,
        },
        'metadata': {
            'num_eval_samples': len(eval_dataset),
            'evaluation_time_sec': inference_time,
        }
    }
    
    # Save metrics
    with open(metrics_dir / 'training_metrics.json', 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    with open(metrics_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    full_results = {'training': training_metrics, 'evaluation': evaluation_results}
    with open(metrics_dir / 'full_results.json', 'w') as f:
        json.dump(full_results, f, indent=2)
    
    # Generate markdown report
    report_lines = [
        "# QLoRA Training Evaluation Report\\n\\n",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n",
        "## Language Modeling Metrics\\n\\n",
        "| Metric | Value |\\n|--------|-------|\\n",
        f"| Train Loss | {evaluation_results['language_modeling']['train_loss']:.4f} |\\n",
        f"| Perplexity | {evaluation_results['language_modeling']['perplexity']:.4f} |\\n\\n",
        "## Generation Quality\\n\\n",
        "| Metric | Value |\\n|--------|-------|\\n",
        f"| BLEU-1 | {gen_metrics['bleu_1']:.2f} |\\n",
        f"| BLEU-2 | {gen_metrics['bleu_2']:.2f} |\\n",
        f"| BLEU-4 | {gen_metrics['bleu_4']:.2f} |\\n",
        f"| ROUGE-1 | {gen_metrics['rouge_1']:.4f} |\\n",
        f"| ROUGE-2 | {gen_metrics['rouge_2']:.4f} |\\n",
        f"| ROUGE-L | {gen_metrics['rouge_l']:.4f} |\\n",
        f"| Exact Match | {gen_metrics['exact_match']:.4f} |\\n",
        f"| Semantic Similarity | {gen_metrics['semantic_similarity']:.4f} |\\n\\n",
        "## Efficiency\\n\\n",
        "| Metric | Value |\\n|--------|-------|\\n",
        f"| Training Time | {evaluation_results['efficiency']['total_training_time_sec']:.2f}s |\\n",
        f"| Inference Time | {evaluation_results['efficiency']['avg_inference_time_ms']:.2f}ms |\\n",
        f"| GPU Memory | {evaluation_results['efficiency']['peak_gpu_memory_mb']:.2f}MB |\\n",
        f"| Trainable % | {evaluation_results['efficiency']['trainable_percentage']:.2f}% |\\n",
    ]
    
    with open(metrics_dir / 'evaluation_report.md', 'w') as f:
        f.writelines(report_lines)
    
    logger.info(f"✓ Metrics saved to {metrics_dir}")
    return evaluation_results

'''

# Insert evaluation functions
new_lines = lines[:insert_pos] + [eval_functions] + lines[insert_pos:]

# Now find and replace _run_simplified_training method
method_start = None
method_end = None
content_str = ''.join(new_lines)

# Simple approach: find the method and replace it
old_method_marker = "def _run_simplified_training(self, config: Dict[str, Any]) -> Dict[str, Any]:"
if old_method_marker not in content_str:
    print("Error: Could not find _run_simplified_training method")
    sys.exit(1)

# Find method boundaries
for i, line in enumerate(new_lines):
    if old_method_marker in line:
        method_start = i
    if method_start and i > method_start and line.strip().startswith("def ") and old_method_marker not in line:
        method_end = i
        break

if not method_start or not method_end:
    print("Error: Could not determine method boundaries")
    sys.exit(1)

print(f"✓ Found method at lines {method_start+1} to {method_end}")

# New enhanced method
new_method = """    def _run_simplified_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Enhanced training with comprehensive evaluation\"\"\"
        import torch
        import time
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
            DataCollatorForLanguageModeling, EvalPrediction
        )
        from peft import LoraConfig, TaskType, get_peft_model
        from datasets import load_dataset
        from pathlib import Path
        
        logger.info("Running enhanced training with comprehensive evaluation...")
        
        # Load model
        model_name = config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=False
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=config.get("lora_r", 32), lora_alpha=config.get("lora_alpha", 64),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]), bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Load and split dataset (80/20)
        data_path = config.get("data_path", "data/curated_seed_data.jsonl")
        dataset = load_dataset('json', data_files=data_path, split='train')
        split_dataset = dataset.train_test_split(test_size=config.get("val_split_ratio", 0.2), seed=42)
        train_dataset, eval_dataset = split_dataset['train'], split_dataset['test']
        logger.info(f"Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        # Tokenize
        def tokenize_function(examples):
            texts = [f"### Instruction:\\\\n{examples['instruction'][i]}\\\\n\\\\n### Output:\\\\n{examples['output'][i]}\\\\n" 
                    for i in range(len(examples['instruction']))]
            return tokenizer(texts, truncation=True, max_length=config.get("max_length", 512))
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)
        
        # Training arguments with evaluation
        output_dir = config.get("output_dir", "models/enhanced-qlora")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.get("num_epochs", 3),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            learning_rate=config.get("learning_rate", 2e-4),
            logging_steps=config.get("logging_steps", 10),
            save_steps=config.get("save_steps", 100),
            max_steps=config.get("max_steps", -1),
            warmup_steps=config.get("warmup_steps", 10),
            evaluation_strategy="steps",
            eval_steps=config.get("eval_steps", 50),
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=f"{output_dir}/logs",
            report_to=["tensorboard"],
            logging_first_step=True,
            fp16=False, bf16=False, remove_unused_columns=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=tokenized_train, eval_dataset=tokenized_eval,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        
        # Train
        logger.info("Starting training...")
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        # Save
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        training_metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", training_time),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "epoch": train_result.metrics.get("epoch", 0),
        }
        
        # Comprehensive evaluation
        try:
            evaluation_results = run_comprehensive_evaluation(
                model, tokenizer, eval_dataset, training_metrics, output_dir
            )
            logger.info("[SUCCESS] Training and evaluation completed!")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            evaluation_results = {}
        
        logger.info(f"[OUTPUT] Model: {output_dir}")
        logger.info(f"[METRICS] Saved to: {output_dir}/metrics/")
        logger.info(f"[REPORT] {output_dir}/metrics/evaluation_report.md")
        
        return {
            "status": "completed", "output_dir": output_dir,
            "training_metrics": training_metrics,
            "evaluation_results": evaluation_results,
            "config": config,
        }
"""

# Replace method
final_lines = new_lines[:method_start] + [new_method + '\n'] + new_lines[method_end:]

# Write updated file
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(final_lines)

print(f"✓ Updated {file_path}")
print(f"✓ Added evaluation functions before class")
print(f"✓ Replaced training method with evaluation")
print("\n✅ All changes applied successfully!")
print(f"Backup: {backup_path}")
