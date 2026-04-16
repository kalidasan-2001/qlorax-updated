#!/usr/bin/env python3
"""
Run a full reproducibility experiment in the isolated module.

Workflow:
1) Run reproducibility training twice (same config/seed)
2) Run deterministic inference for both runs on fixed prompts
3) Call compare_runs.py
4) Save a final summary report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from repro_utils import set_reproducible


@dataclass
class CmdResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str


def run_cmd(cmd: List[str], check: bool = True) -> CmdResult:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    result = CmdResult(command=cmd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    if check and proc.returncode != 0:
        msg = (
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        raise RuntimeError(msg)
    return result


def load_prompts(prompts_file: Path) -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = []
    with prompts_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = str(obj.get("id", f"p{i:02d}"))
            prompt = str(obj.get("prompt") or obj.get("instruction") or "").strip()
            if not prompt:
                raise ValueError(f"Prompt missing text at line {i} in {prompts_file}")
            prompts.append({"id": pid, "prompt": prompt})
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")
    return prompts


def run_deterministic_inference(run_dir: Path, prompts: List[Dict[str, str]], max_new_tokens: int, seed: int) -> Dict[str, Any]:
    """Load trained model from run_dir/model and generate deterministic outputs."""
    set_reproducible(seed=seed, deterministic=True, cpu_only=True, num_threads=1)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = run_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try direct load first; if adapter-only model, fallback to PEFT adapter load.
    model = None
    load_mode = "direct"
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    except Exception:
        adapter_cfg = model_dir / "adapter_config.json"
        if not adapter_cfg.exists():
            raise
        from peft import PeftModel

        cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        base_name = cfg.get("base_model_name_or_path")
        if not base_name:
            raise ValueError(f"adapter_config.json missing base_model_name_or_path in {adapter_cfg}")
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(base, str(model_dir))
        load_mode = "peft_adapter"

    model.to("cpu")
    model.eval()

    output_path = run_dir / "fixed_prompt_outputs.jsonl"
    rows_written = 0

    with output_path.open("w", encoding="utf-8") as out:
        for p in prompts:
            prompt_text = p["prompt"]
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True)
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            # Keep only generated continuation when possible
            completion = decoded[len(prompt_text):].strip() if decoded.startswith(prompt_text) else decoded.strip()

            row = {
                "id": p["id"],
                "prompt": prompt_text,
                "output": completion,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1

    return {
        "run_dir": str(run_dir),
        "inference_file": str(output_path),
        "rows_written": rows_written,
        "model_load_mode": load_mode,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full reproducibility experiment orchestrator")
    p.add_argument(
        "--config",
        default="reproducibility/configs/repro_config.yaml",
        help="Path to reproducibility config YAML",
    )
    p.add_argument(
        "--experiment-name",
        default=None,
        help="Optional explicit experiment name (default: timestamped)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Deterministic inference generation length",
    )
    p.add_argument(
        "--summary-path",
        default=None,
        help="Optional explicit final summary JSON path",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    exp_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    seed = int(exp_cfg.get("seed", 42))
    output_root = Path(train_cfg.get("output_root", "reproducibility/outputs"))
    manifests_root = Path("reproducibility/manifests")
    output_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"full_repro_{stamp}_seed{seed}"

    run_a = f"{experiment_name}_run01"
    run_b = f"{experiment_name}_run02"

    prompts_file = Path(data_cfg.get("prompts_file", "reproducibility/configs/fixed_prompts.jsonl"))
    prompts = load_prompts(prompts_file)

    summary: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "config_path": str(cfg_path),
        "seed": seed,
        "runs": {"run_a": run_a, "run_b": run_b},
        "steps": {},
        "compare": {},
        "status": "started",
    }

    # Step 1: Train twice (same config/seed)
    cmd_a = [
        sys.executable,
        "reproducibility/scripts/run_repro_train.py",
        "--config",
        str(cfg_path),
        "--run-name",
        run_a,
    ]
    cmd_b = [
        sys.executable,
        "reproducibility/scripts/run_repro_train.py",
        "--config",
        str(cfg_path),
        "--run-name",
        run_b,
    ]

    res_a = run_cmd(cmd_a, check=True)
    res_b = run_cmd(cmd_b, check=True)

    summary["steps"]["train_run_a"] = {"command": cmd_a, "returncode": res_a.returncode}
    summary["steps"]["train_run_b"] = {"command": cmd_b, "returncode": res_b.returncode}

    run_a_dir = output_root / run_a
    run_b_dir = output_root / run_b

    # Step 2: Deterministic inference for both runs
    inf_a = run_deterministic_inference(run_a_dir, prompts, max_new_tokens=args.max_new_tokens, seed=seed)
    inf_b = run_deterministic_inference(run_b_dir, prompts, max_new_tokens=args.max_new_tokens, seed=seed)

    summary["steps"]["inference_run_a"] = inf_a
    summary["steps"]["inference_run_b"] = inf_b

    # Step 3: Compare runs
    compare_report = manifests_root / f"compare_{run_a}_vs_{run_b}.json"
    compare_cmd = [
        sys.executable,
        "reproducibility/scripts/compare_runs.py",
        "--run-a",
        run_a,
        "--run-b",
        run_b,
        "--outputs-root",
        str(output_root),
        "--report",
        str(compare_report),
    ]

    compare_res = run_cmd(compare_cmd, check=False)
    summary["steps"]["compare_runs"] = {
        "command": compare_cmd,
        "returncode": compare_res.returncode,
        "report": str(compare_report),
    }

    compare_json = {}
    if compare_report.exists():
        compare_json = json.loads(compare_report.read_text(encoding="utf-8"))
    summary["compare"] = compare_json

    # Step 4: Final summary report
    final_summary_path = Path(args.summary_path) if args.summary_path else manifests_root / f"final_summary_{experiment_name}.json"
    summary["final_summary_path"] = str(final_summary_path)

    if compare_res.returncode == 0:
        summary["status"] = "success_match"
    elif compare_res.returncode == 2:
        summary["status"] = "completed_with_mismatches"
    else:
        summary["status"] = "compare_error"

    final_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print("Full Reproducibility Experiment Complete")
    print("=" * 72)
    print(f"Experiment: {experiment_name}")
    print(f"Run A: {run_a_dir}")
    print(f"Run B: {run_b_dir}")
    print(f"Compare report: {compare_report}")
    print(f"Final summary: {final_summary_path}")
    print(f"Status: {summary['status']}")

    # Propagate mismatch as non-zero (thesis/CI friendly)
    return 0 if compare_res.returncode == 0 else compare_res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
