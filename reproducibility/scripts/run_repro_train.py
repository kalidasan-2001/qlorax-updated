#!/usr/bin/env python3
"""
Standalone reproducibility-first training script for the isolated module.

Design goals:
- Self-contained (does not depend on main training pipeline)
- Deterministic by default
- CPU-friendly lightweight LoRA experiment
- Writes all artifacts under reproducibility/outputs/<run_name>/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from repro_utils import save_environment_metadata, set_reproducible


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _build_text_sample(row: Dict[str, Any]) -> str:
    """Normalize common dataset schemas into a single training text format."""
    if "instruction" in row and "output" in row:
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", "")).strip()
        return f"### Instruction:\n{instruction}\n\n### Output:\n{output}"

    if "input" in row and "output" in row:
        input_text = str(row.get("input", "")).strip()
        output = str(row.get("output", "")).strip()
        return f"### Input:\n{input_text}\n\n### Output:\n{output}"

    if "text" in row:
        return str(row["text"])

    # Last resort: deterministic string snapshot
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_hash_manifest(run_dir: Path) -> Dict[str, str]:
    """Hash key artifacts if present."""
    candidates = [
        run_dir / "config" / "repro_config.used.yaml",
        run_dir / "metadata" / "environment.json",
        run_dir / "metadata" / "run_metadata.json",
        run_dir / "model" / "adapter_config.json",
        run_dir / "model" / "adapter_model.safetensors",
        run_dir / "model" / "adapter_model.bin",
        run_dir / "model" / "training_args.bin",
        run_dir / "model" / "tokenizer_config.json",
    ]

    hashes: Dict[str, str] = {}
    for p in candidates:
        if p.exists() and p.is_file():
            hashes[str(p.relative_to(run_dir))] = _sha256_file(p)

    # Also include any trainer_state if present
    for p in run_dir.rglob("trainer_state.json"):
        hashes[str(p.relative_to(run_dir))] = _sha256_file(p)

    return hashes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone reproducibility-first LoRA training")
    p.add_argument(
        "--config",
        default="reproducibility/configs/repro_config.yaml",
        help="Path to reproducibility YAML config",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit run name (default: auto timestamp)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Optional model override. Default is CPU-friendly sshleifer/tiny-gpt2",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="Max number of training samples (small for CPU reproducibility)",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenization max length",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max training steps (keeps CPU runs short/repeatable)",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for lightweight experiment",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Resolve core config with safe defaults
    exp_cfg = cfg.get("experiment", {})
    det_cfg = cfg.get("determinism", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    seed = int(exp_cfg.get("seed", 42))
    deterministic = bool(det_cfg.get("use_deterministic_algorithms", True))
    cpu_only = str(train_cfg.get("device", "cpu")).lower() == "cpu"
    num_threads = int(det_cfg.get("num_threads", 1))

    # Repro setup first
    repro_summary = set_reproducible(
        seed=seed,
        deterministic=deterministic,
        cpu_only=cpu_only,
        num_threads=num_threads,
    )

    # Imports after reproducibility setup to honor env/thread settings early
    import random
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # Build run directory
    output_root = Path(train_cfg.get("output_root", "reproducibility/outputs"))
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"repro_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}"
    run_dir = output_root / run_name
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    shutil.copy2(cfg_path, run_dir / "config" / "repro_config.used.yaml")

    # Save environment metadata
    env_meta_path = save_environment_metadata(run_dir / "metadata" / "environment.json")

    # Load dataset (or deterministic fallback tiny dataset)
    train_file = Path(data_cfg.get("train_file", "data/splits/hybrid_70_30/train.jsonl"))
    if train_file.exists():
        rows = _read_jsonl(train_file)
    else:
        rows = [
            {"instruction": f"Define reproducibility concept {i}", "output": f"Reproducibility explanation {i}"}
            for i in range(1, 33)
        ]

    # Deterministic subset selection
    rows = rows[: max(1, args.max_samples)]

    texts = [_build_text_sample(r) for r in rows]
    ds = Dataset.from_list([{"text": t} for t in texts])

    # CPU-friendly default model for thesis reproducibility loops
    # (You can override with --model or a config model value)
    model_name = args.model or cfg.get("model", {}).get("repro_model", "sshleifer/tiny-gpt2")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to("cpu")

    # Minimal LoRA config (CPU-feasible)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(train_cfg.get("lora_r", 4)),
        lora_alpha=int(train_cfg.get("lora_alpha", 16)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.0)),
        target_modules=train_cfg.get("target_modules", ["c_attn", "c_proj"]),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=int(train_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=args.learning_rate,
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_steps=int(train_cfg.get("save_steps", 10)),
        save_total_limit=int(train_cfg.get("save_total_limit", 1)),
        report_to=[],
        seed=seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    train_result = trainer.train()

    # Save model/tokenizer artifacts
    trainer.save_model(str(run_dir / "model"))
    tokenizer.save_pretrained(str(run_dir / "model"))

    # Save run metadata
    train_metrics = dict(train_result.metrics) if hasattr(train_result, "metrics") else {}
    metadata = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "seed": seed,
        "deterministic": deterministic,
        "cpu_only": cpu_only,
        "num_threads": num_threads,
        "model_name": model_name,
        "num_samples": len(rows),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "train_file": str(train_file),
        "environment_metadata_path": str(env_meta_path),
        "repro_setup": repro_summary,
        "train_metrics": train_metrics,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    (run_dir / "metadata" / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save hashes for key artifacts
    hashes = _compute_hash_manifest(run_dir)
    (run_dir / "metadata" / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[DONE] Reproducibility training run complete")
    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] samples={len(rows)} model={model_name}")
    print(f"[INFO] artifacts_hashed={len(hashes)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
