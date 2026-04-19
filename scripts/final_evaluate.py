#!/usr/bin/env python3
"""
Final thesis evaluation script for QLoRAx.

Evaluates a PEFT adapter model on JSONL eval data with fields:
- instruction
- output

Outputs:
- <output-dir>/detailed_results.json
- <output-dir>/predictions.json
- <output-dir>/evaluation_report.md
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftModel
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final thesis model evaluation")
    parser.add_argument("--model", required=True, help="PEFT adapter directory")
    parser.add_argument(
        "--eval-data", required=True, help="Path to eval JSONL with instruction/output"
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    parser.add_argument(
        "--baseline-json",
        help="Optional baseline detailed_results.json for regression comparison",
    )
    return parser.parse_args()


def validate_paths(model_dir: Path, eval_data: Path):
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path not found: {model_dir}")
    if not eval_data.exists():
        raise FileNotFoundError(f"Eval data path not found: {eval_data}")

    adapter_cfg = model_dir / "adapter_config.json"
    if not adapter_cfg.exists():
        raise FileNotFoundError(
            f"Expected PEFT adapter_config.json in model dir: {adapter_cfg}"
        )


def load_eval_data(eval_data_path: Path) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []

    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

            if "instruction" not in item or "output" not in item:
                raise ValueError(
                    f"Missing required fields at line {line_num}. "
                    f"Required: instruction/output. Found keys: {list(item.keys())}"
                )

            samples.append(
                {
                    "instruction": str(item["instruction"]),
                    "output": str(item["output"]),
                }
            )

    if not samples:
        raise ValueError("Eval dataset is empty after parsing")

    return samples


def load_model_and_tokenizer(model_dir: Path):
    with open(model_dir / "adapter_config.json", "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("adapter_config.json missing base_model_name_or_path")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float32, device_map="cpu"
        )

    model = PeftModel.from_pretrained(base_model, str(model_dir))
    model.eval()
    return model, tokenizer


def generate_prediction(model, tokenizer, instruction: str, max_new_tokens: int = 128):
    prompt = f"### Instruction:\n{instruction}\n\n### Output:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    try:
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
    except StopIteration:
        pass

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000.0

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = decoded.split("### Output:")[-1].strip()
    return prediction, latency_ms


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(predictions)


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    smoothing = SmoothingFunction().method1
    bleu1, bleu2, bleu4 = [], [], []

    for p, r in zip(predictions, references):
        p_tokens = nltk.word_tokenize(p.lower())
        r_tokens = [nltk.word_tokenize(r.lower())]

        bleu1.append(
            sentence_bleu(r_tokens, p_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        )
        bleu2.append(
            sentence_bleu(r_tokens, p_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        )
        bleu4.append(
            sentence_bleu(
                r_tokens,
                p_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )
        )

    return {
        "bleu_1": float(np.mean(bleu1) * 100.0),
        "bleu_2": float(np.mean(bleu2) * 100.0),
        "bleu_4": float(np.mean(bleu4) * 100.0),
    }


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []

    for p, r in zip(predictions, references):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    return {
        "rouge_1": float(np.mean(r1)),
        "rouge_2": float(np.mean(r2)),
        "rouge_l": float(np.mean(rl)),
    }


def compute_semantic_similarity(predictions: List[str], references: List[str]) -> float:
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    pred_embeddings = semantic_model.encode(predictions, show_progress_bar=False)
    ref_embeddings = semantic_model.encode(references, show_progress_bar=False)

    sims = []
    for pe, re in zip(pred_embeddings, ref_embeddings):
        sims.append(cosine_similarity([pe], [re])[0][0])

    return float(np.mean(sims))


def _build_scoring_example(tokenizer, instruction: str, reference: str):
    prompt = f"### Instruction:\n{instruction}\n\n### Output:\n"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    reference_ids = tokenizer(reference, add_special_tokens=False)["input_ids"]

    if not reference_ids:
        raise ValueError("reference output tokenized to zero tokens")

    input_ids = list(prompt_ids) + list(reference_ids)
    labels = [-100] * len(prompt_ids) + list(reference_ids)

    if tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    if len(labels) != len(input_ids):
        raise ValueError("failed to align scoring labels")

    return input_ids, labels, len(reference_ids) + (1 if tokenizer.eos_token_id is not None else 0)


def compute_eval_loss(model, tokenizer, samples: List[Dict[str, str]]) -> Tuple[float, int, int]:
    total_loss = 0.0
    total_tokens = 0
    scored = 0
    skipped = 0

    for sample in samples:
        try:
            input_ids, labels, ref_tokens = _build_scoring_example(
                tokenizer, sample["instruction"], sample["output"]
            )
            if ref_tokens <= 0:
                skipped += 1
                continue

            inputs = {
                "input_ids": torch.tensor([input_ids]),
                "labels": torch.tensor([labels]),
            }

            try:
                model_device = next(model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            except StopIteration:
                pass

            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs.loss
            if loss is None:
                skipped += 1
                continue

            total_loss += float(loss.item()) * ref_tokens
            total_tokens += ref_tokens
            scored += 1
        except Exception as exc:
            skipped += 1
            print(f"[WARN] Skipping loss score for sample: {exc}")
            continue

    if total_tokens == 0:
        raise RuntimeError("Unable to compute eval loss: no samples could be scored")

    return total_loss / total_tokens, scored, skipped


def load_baseline_results(baseline_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if not baseline_json:
        return None

    path = Path(baseline_json)
    if not path.exists():
        raise FileNotFoundError(f"Baseline JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_metric(current: float, baseline: float, higher_is_better: bool) -> Dict[str, Any]:
    delta = current - baseline
    if abs(delta) < 1e-12:
        status = "unchanged"
    else:
        improved = delta > 0 if higher_is_better else delta < 0
        status = "improved" if improved else "degraded"

    percent_change = None
    if baseline != 0:
        percent_change = (delta / baseline) * 100.0

    return {
        "baseline": baseline,
        "current": current,
        "delta": delta,
        "percent_change": percent_change,
        "status": status,
        "higher_is_better": higher_is_better,
    }


def build_regression_summary(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    specs = {
        "semantic_similarity": True,
        "bleu_4": True,
        "rouge_l": True,
        "perplexity": False,
        "avg_inference_time_ms": False,
    }

    metrics: Dict[str, Dict[str, Any]] = {}
    improved = degraded = unchanged = missing = 0

    for metric, higher_is_better in specs.items():
        current_value = current.get(metric)
        baseline_value = baseline.get(metric)
        if current_value is None or baseline_value is None:
            metrics[metric] = {"status": "missing", "current": current_value, "baseline": baseline_value}
            missing += 1
            continue

        comparison = compare_metric(float(current_value), float(baseline_value), higher_is_better)
        metrics[metric] = comparison
        if comparison["status"] == "improved":
            improved += 1
        elif comparison["status"] == "degraded":
            degraded += 1
        else:
            unchanged += 1

    if degraded and improved:
        overall_status = "mixed"
    elif degraded:
        overall_status = "degraded"
    elif improved:
        overall_status = "improved"
    else:
        overall_status = "unchanged"

    return {
        "baseline_path": current.get("baseline_path"),
        "overall_status": overall_status,
        "counts": {
            "improved": improved,
            "degraded": degraded,
            "unchanged": unchanged,
            "missing": missing,
        },
        "metrics": metrics,
    }


def save_outputs(
    output_dir: Path,
    results: Dict[str, Any],
    predictions_payload: List[Dict[str, str]],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = output_dir / "detailed_results.json"
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions_payload, f, indent=2, ensure_ascii=False)

    report_path = output_dir / "evaluation_report.md"
    report = f"""# Final Thesis Evaluation Report

Generated: {datetime.now().isoformat()}

## Inputs
- Model: {results['model_path']}
- Eval data: {results['eval_data_path']}
- Samples: {results['num_eval_samples']}
"""
    if results.get("baseline_path"):
        report += f"- Baseline: {results['baseline_path']}\n"
    report += f"""
## Metrics
- Exact Match: {results['exact_match']:.4f}
- Semantic Similarity: {results['semantic_similarity']:.4f}
- BLEU-1: {results['bleu_1']:.4f}
- BLEU-2: {results['bleu_2']:.4f}
- BLEU-4: {results['bleu_4']:.4f}
- ROUGE-1: {results['rouge_1']:.4f}
- ROUGE-2: {results['rouge_2']:.4f}
- ROUGE-L: {results['rouge_l']:.4f}
- Eval Loss: {results['eval_loss']:.4f}
- Perplexity: {results['perplexity']:.4f}
- Avg Inference Time (ms): {results['average_inference_time_ms']:.2f}
"""
    regression_summary = results.get("regression_summary")
    if regression_summary:
        report += f"""
## Regression Check
- Baseline Status: {regression_summary['overall_status']}
- Improved: {regression_summary['counts']['improved']}
- Degraded: {regression_summary['counts']['degraded']}
- Unchanged: {regression_summary['counts']['unchanged']}
- Missing: {regression_summary['counts']['missing']}

| Metric | Baseline | Current | Change | Status |
| --- | ---: | ---: | ---: | --- |
"""
        for metric_name, metric_data in regression_summary["metrics"].items():
            baseline = metric_data.get("baseline")
            current = metric_data.get("current")
            delta = metric_data.get("delta")
            status = metric_data.get("status")
            if baseline is None or current is None or delta is None:
                report += f"| {metric_name} | n/a | n/a | n/a | {status} |\n"
            else:
                report += f"| {metric_name} | {baseline:.4f} | {current:.4f} | {delta:+.4f} | {status} |\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)


def main():
    args = parse_args()

    model_dir = Path(args.model)
    eval_data_path = Path(args.eval_data)
    output_dir = Path(args.output_dir)

    validate_paths(model_dir, eval_data_path)

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Use a new output path to avoid overwriting results."
        )

    print("[FINAL EVAL]")
    print(f"  model={model_dir}")
    print(f"  eval_data={eval_data_path}")
    print(f"  output_dir={output_dir}")

    print("[STAGE] loading eval data...")
    samples = load_eval_data(eval_data_path)
    print(f"[STAGE] loaded eval samples: {len(samples)}")
    print("[STAGE] loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_dir)
    print("[STAGE] model and tokenizer loaded")

    predictions_payload: List[Dict[str, str]] = []
    predictions: List[str] = []
    references: List[str] = []
    latencies: List[float] = []

    print("[STAGE] generating predictions...")

    for idx, sample in enumerate(samples):
        pred, latency = generate_prediction(model, tokenizer, sample["instruction"])
        predictions.append(pred)
        references.append(sample["output"])
        latencies.append(latency)

        predictions_payload.append(
            {
                "id": idx,
                "instruction": sample["instruction"],
                "reference": sample["output"],
                "prediction": pred,
                "inference_time_ms": latency,
            }
        )

    bleu = compute_bleu(predictions, references)
    rouge = compute_rouge(predictions, references)
    eval_loss, scored_samples, skipped_samples = compute_eval_loss(model, tokenizer, samples)
    perplexity = float(math.exp(eval_loss))

    results = {
        "exact_match": compute_exact_match(predictions, references),
        "semantic_similarity": compute_semantic_similarity(predictions, references),
        **bleu,
        **rouge,
        "eval_loss": float(eval_loss),
        "perplexity": perplexity,
        "average_inference_time_ms": float(np.mean(latencies)),
        "avg_inference_time_ms": float(np.mean(latencies)),
        "model_path": str(model_dir),
        "eval_data_path": str(eval_data_path),
        "num_eval_samples": len(samples),
        "num_eval_samples_scored": scored_samples,
        "num_eval_samples_skipped": skipped_samples,
        "timestamp": datetime.now().isoformat(),
    }

    baseline_results = load_baseline_results(args.baseline_json)
    if baseline_results is not None:
        results["baseline_path"] = str(Path(args.baseline_json))
        results["regression_summary"] = build_regression_summary(results, baseline_results)

    save_outputs(output_dir, results, predictions_payload)
    print("Final evaluation complete")


if __name__ == "__main__":
    main()
