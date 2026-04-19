"""Main orchestrator for the isolated RQ4 evaluation gate.

This script coordinates a controlled CI/CD-style gate that compares a candidate
LoRA release against a blessed baseline and emits thesis-friendly manifests.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from rq4_evaluation.scripts.compare_rq4_results import compare_rq4_results  # type: ignore
    from rq4_evaluation.scripts.rq4_utils import (  # type: ignore
        ensure_dir,
        load_json,
        load_yaml,
        read_jsonl,
        safe_float,
        timestamp_now,
        validate_required_files,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:
    from compare_rq4_results import compare_rq4_results
    from rq4_utils import (
        ensure_dir,
        load_json,
        load_yaml,
        read_jsonl,
        safe_float,
        timestamp_now,
        validate_required_files,
        write_json,
        write_jsonl,
    )


PASS = "pass"
DEGRADED_RELEASE = "degraded_release"
RUNTIME_OR_CONFIG_ERROR = "runtime_or_config_error"

EXIT_PASS = 0
EXIT_DEGRADED = 2
EXIT_RUNTIME_ERROR = 10


def _truncate(text: str | None, max_chars: int = 1200) -> str:
    """Return a trimmed summary string for logs/manifests."""
    if not text:
        return ""
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars]}... [truncated]"


def _resolve_path(value: str | Path | None) -> Path | None:
    """Resolve path-like input to absolute path using current working directory."""
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (Path.cwd() / p)


def parse_args() -> argparse.Namespace:
    """Parse command-line args for the RQ4 gate orchestrator."""
    parser = argparse.ArgumentParser(description="Run isolated RQ4 candidate-vs-baseline gate.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("rq4_evaluation/configs/rq4_gate_ci.yaml"),
        help="Path to RQ4 gate config YAML.",
    )
    parser.add_argument(
        "--candidate-model",
        type=Path,
        default=None,
        help="Candidate LoRA adapter directory for evaluator execution.",
    )
    parser.add_argument(
        "--candidate-metrics",
        type=Path,
        default=None,
        help="Optional override for candidate metrics JSON path.",
    )
    parser.add_argument(
        "--candidate-outputs",
        type=Path,
        default=None,
        help="Optional override for candidate outputs JSONL path.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluator execution and use existing candidate artifacts.",
    )
    return parser.parse_args()


def load_gate_config(config_path: Path) -> dict[str, Any]:
    """Load and minimally validate top-level RQ4 gate config."""
    data = load_yaml(config_path)
    if not isinstance(data, dict):
        raise ValueError(f"Gate config must be a mapping: {config_path}")
    return data


def preflight_checks(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    """Run minimal preflight checks and resolve core path dependencies."""
    required_top_level = ("paths", "evaluator", "deterministic")
    missing = [k for k in required_top_level if k not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    paths_cfg = config.get("paths")
    if not isinstance(paths_cfg, dict):
        raise ValueError("Config key 'paths' must be a mapping")

    for required_key in (
        "baseline_config",
        "thresholds_config",
        "outputs_dir",
        "manifests_dir",
    ):
        if required_key not in paths_cfg:
            raise ValueError(f"Config missing paths.{required_key}")

    resolved: dict[str, Path] = {
        "config_path": config_path.resolve(),
        "baseline_config": _resolve_path(paths_cfg["baseline_config"]).resolve(),
        "thresholds_config": _resolve_path(paths_cfg["thresholds_config"]).resolve(),
        "outputs_dir": _resolve_path(paths_cfg["outputs_dir"]).resolve(),
        "manifests_dir": _resolve_path(paths_cfg["manifests_dir"]).resolve(),
    }

    validate_required_files(
        base_path=Path("/"),
        filenames=[
            str(resolved["baseline_config"]),
            str(resolved["thresholds_config"]),
        ],
    )

    ensure_dir(resolved["outputs_dir"])
    ensure_dir(resolved["manifests_dir"])
    return resolved


def validate_candidate_artifacts(candidate_model: Path) -> None:
    """Validate candidate adapter directory presence for evaluator execution."""
    if not candidate_model.exists():
        raise FileNotFoundError(f"Candidate model path not found: {candidate_model}")
    adapter_cfg = candidate_model / "adapter_config.json"
    if not adapter_cfg.is_file():
        raise FileNotFoundError(f"Expected candidate adapter_config.json at: {adapter_cfg}")


def _build_eval_output_dir(outputs_dir: Path) -> Path:
    """Build a unique, non-existent directory for evaluator raw outputs.

    scripts/final_evaluate.py fails if --output-dir already exists, so the
    orchestrator must pass a fresh path each run.
    """
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    candidate = outputs_dir / f"raw_final_evaluate_{stamp}"

    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = outputs_dir / f"raw_final_evaluate_{stamp}_{suffix:02d}"

    return candidate


def _run_final_evaluate_import(
    candidate_model: Path,
    eval_data_path: Path,
    raw_output_dir: Path,
) -> dict[str, Any] | None:
    """Try import-based evaluator execution if reusable function is available.

    Returns None when no safe reusable function signature exists.
    """
    try:
        module = importlib.import_module("scripts.final_evaluate")
    except ModuleNotFoundError:
        # Common when running as a script and repo root is not import-packaged.
        return None

    for fn_name in ("run_evaluation", "evaluate_model", "evaluate"):
        fn = getattr(module, fn_name, None)
        if not callable(fn):
            continue

        signature = inspect.signature(fn)
        params = set(signature.parameters.keys())

        kwargs: dict[str, Any] = {}
        if {"model", "eval_data", "output_dir"}.issubset(params):
            kwargs = {
                "model": str(candidate_model),
                "eval_data": str(eval_data_path),
                "output_dir": str(raw_output_dir),
            }
        elif {"model_dir", "eval_data_path", "output_dir"}.issubset(params):
            kwargs = {
                "model_dir": candidate_model,
                "eval_data_path": eval_data_path,
                "output_dir": raw_output_dir,
            }
        else:
            continue

        fn(**kwargs)
        return {
            "ok": True,
            "method_used": "import",
            "return_code": 0,
            "stdout_summary": f"Imported scripts.final_evaluate.{fn_name} executed successfully.",
            "stderr_summary": "",
            "error_message": None,
        }

    return None


def execute_evaluation(
    *,
    candidate_model: Path,
    eval_data_path: Path,
    outputs_dir: Path,
    metric_name: str,
) -> dict[str, Any]:
    """Execute candidate evaluation with import-first/subprocess-fallback strategy.

    Returns structured execution metadata plus normalized candidate artifact paths.
    """
    ensure_dir(outputs_dir)
    raw_output_dir = _build_eval_output_dir(outputs_dir)

    candidate_metrics_path = outputs_dir / "candidate_metrics.json"
    candidate_outputs_path = outputs_dir / "candidate_outputs.jsonl"

    method_used = "subprocess"
    return_code: int | None = None
    stdout_summary = ""
    stderr_summary = ""
    error_message: str | None = None

    try:
        import_result = _run_final_evaluate_import(candidate_model, eval_data_path, raw_output_dir)
        if import_result is not None:
            method_used = "import"
            return_code = int(import_result.get("return_code", 0))
            stdout_summary = str(import_result.get("stdout_summary", ""))
            stderr_summary = str(import_result.get("stderr_summary", ""))
        else:
            eval_script = (Path.cwd() / "scripts" / "final_evaluate.py").resolve()
            if not eval_script.is_file():
                raise FileNotFoundError(f"Expected evaluator script not found: {eval_script}")

            cmd = [
                sys.executable,
                str(eval_script),
                "--model",
                str(candidate_model),
                "--eval-data",
                str(eval_data_path),
                "--output-dir",
                str(raw_output_dir),
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return_code = completed.returncode
            stdout_summary = _truncate(completed.stdout)
            stderr_summary = _truncate(completed.stderr)

            if completed.returncode != 0:
                return {
                    "ok": False,
                    "method_used": "subprocess",
                    "metrics_path": str(candidate_metrics_path),
                    "outputs_path": str(candidate_outputs_path),
                    "return_code": completed.returncode,
                    "error_message": "Evaluator subprocess exited with non-zero status.",
                    "stdout_summary": stdout_summary,
                    "stderr_summary": stderr_summary,
                }

        detailed_results_path = raw_output_dir / "detailed_results.json"
        predictions_path = raw_output_dir / "predictions.json"
        if not detailed_results_path.is_file() or not predictions_path.is_file():
            return {
                "ok": False,
                "method_used": method_used,
                "metrics_path": str(candidate_metrics_path),
                "outputs_path": str(candidate_outputs_path),
                "return_code": return_code,
                "error_message": "Expected evaluator output files were not produced.",
                "stdout_summary": stdout_summary,
                "stderr_summary": stderr_summary,
            }

        detailed = load_json(detailed_results_path)
        predictions = load_json(predictions_path)

        metric_value = safe_float(detailed.get(metric_name)) if isinstance(detailed, dict) else None
        source_metric = metric_name
        if metric_value is None and isinstance(detailed, dict):
            # Minimal safe fallback when metric_name is absent in final_evaluate outputs.
            for fallback_metric in ("exact_match", "semantic_similarity", "bleu_4", "rouge_l"):
                fallback_value = safe_float(detailed.get(fallback_metric))
                if fallback_value is not None:
                    metric_value = fallback_value
                    source_metric = fallback_metric
                    break

        if metric_value is None:
            return {
                "ok": False,
                "method_used": method_used,
                "metrics_path": str(candidate_metrics_path),
                "outputs_path": str(candidate_outputs_path),
                "return_code": return_code,
                "error_message": f"Could not extract numeric metric '{metric_name}' from evaluator outputs.",
                "stdout_summary": stdout_summary,
                "stderr_summary": stderr_summary,
            }

        metrics_payload = {
            "baseline_id": "candidate_current_run",
            "metric_name": metric_name,
            "metric_value": metric_value,
            "source_metric": source_metric,
            "sample_count": int(detailed.get("num_eval_samples", 0)) if isinstance(detailed, dict) else 0,
            "evaluation_mode": "cpu_deterministic",
            "notes": "Generated by rq4 gate wrapper from scripts/final_evaluate.py outputs.",
        }
        write_json(candidate_metrics_path, metrics_payload)

        normalized_rows: list[dict[str, Any]] = []
        if not isinstance(predictions, list):
            return {
                "ok": False,
                "method_used": method_used,
                "metrics_path": str(candidate_metrics_path),
                "outputs_path": str(candidate_outputs_path),
                "return_code": return_code,
                "error_message": "Predictions payload is not a list; cannot normalize outputs.",
                "stdout_summary": stdout_summary,
                "stderr_summary": stderr_summary,
            }

        for idx, item in enumerate(predictions):
            if not isinstance(item, dict):
                continue
            prediction_text = str(item.get("prediction", ""))
            normalized_rows.append(
                {
                    "id": str(item.get("id", idx)),
                    "prompt": str(item.get("instruction", "")),
                    "prediction": prediction_text,
                    "label": str(item.get("reference", "")),
                    "valid": bool(prediction_text.strip()),
                }
            )

        write_jsonl(candidate_outputs_path, normalized_rows)

        return {
            "ok": True,
            "method_used": method_used,
            "metrics_path": str(candidate_metrics_path),
            "outputs_path": str(candidate_outputs_path),
            "return_code": return_code,
            "error_message": error_message,
            "stdout_summary": stdout_summary,
            "stderr_summary": stderr_summary,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "method_used": method_used,
            "metrics_path": str(candidate_metrics_path),
            "outputs_path": str(candidate_outputs_path),
            "return_code": return_code,
            "error_message": f"{type(exc).__name__}: {exc}",
            "stdout_summary": stdout_summary,
            "stderr_summary": _truncate(traceback.format_exc()),
        }


def map_classification_to_exit_code(classification: str) -> int:
    """Map comparison classification to process exit code contract."""
    if classification == PASS:
        return EXIT_PASS
    if classification == DEGRADED_RELEASE:
        return EXIT_DEGRADED
    return EXIT_RUNTIME_ERROR


def main() -> int:
    """Run the isolated RQ4 gate orchestration flow."""
    start_ts = timestamp_now()
    args = parse_args()

    summary: dict[str, Any] = {
        "timestamp_start_utc": start_ts,
        "classification": RUNTIME_OR_CONFIG_ERROR,
        "reasons": [],
    }

    try:
        config_path = _resolve_path(args.config)
        if config_path is None:
            raise ValueError("Config path is required")

        config = load_gate_config(config_path)
        resolved = preflight_checks(config, config_path)

        outputs_dir = resolved["outputs_dir"]
        manifests_dir = resolved["manifests_dir"]

        baseline_cfg = load_yaml(resolved["baseline_config"])
        if not isinstance(baseline_cfg, dict):
            raise ValueError("Baseline config must be a mapping")

        baseline_paths = baseline_cfg.get("paths", {}) if isinstance(baseline_cfg, dict) else {}
        if not isinstance(baseline_paths, dict):
            raise ValueError("Baseline config 'paths' must be a mapping")

        baseline_metrics_path = _resolve_path(baseline_paths.get("metrics"))
        baseline_outputs_path = _resolve_path(baseline_paths.get("outputs"))

        if baseline_metrics_path is None or baseline_outputs_path is None:
            raise ValueError("Baseline config missing required paths.metrics or paths.outputs")

        validate_required_files(
            base_path=Path("/"),
            filenames=[str(baseline_metrics_path), str(baseline_outputs_path)],
        )

        thresholds = load_yaml(resolved["thresholds_config"])
        metric_gate = thresholds.get("metric_gate", {}) if isinstance(thresholds, dict) else {}
        metric_name = str(metric_gate.get("primary_metric_name", ""))
        if not metric_name:
            raise ValueError("Threshold config missing metric_gate.primary_metric_name")

        paths_cfg = config.get("paths", {})
        if not isinstance(paths_cfg, dict):
            raise ValueError("Config key paths must be a mapping")
        candidate_cfg = paths_cfg.get("candidate_artifacts", {})
        if not isinstance(candidate_cfg, dict):
            raise ValueError("Config key paths.candidate_artifacts must be a mapping")

        default_candidate_metrics = _resolve_path(candidate_cfg.get("metrics"))
        default_candidate_outputs = _resolve_path(candidate_cfg.get("outputs"))
        if default_candidate_metrics is None or default_candidate_outputs is None:
            raise ValueError("Config missing candidate artifact paths for metrics/outputs")

        candidate_metrics_path = (_resolve_path(args.candidate_metrics) or default_candidate_metrics).resolve()
        candidate_outputs_path = (_resolve_path(args.candidate_outputs) or default_candidate_outputs).resolve()

        eval_wrapper_result: dict[str, Any] = {
            "ok": True,
            "method_used": "none",
            "metrics_path": str(candidate_metrics_path),
            "outputs_path": str(candidate_outputs_path),
            "return_code": None,
            "error_message": None,
            "stdout_summary": "",
            "stderr_summary": "",
        }

        if args.skip_eval:
            validate_required_files(
                base_path=Path("/"),
                filenames=[str(candidate_metrics_path), str(candidate_outputs_path)],
            )
        else:
            candidate_model = _resolve_path(args.candidate_model)
            if candidate_model is None:
                raise ValueError("--candidate-model is required unless --skip-eval is used")
            candidate_model = candidate_model.resolve()
            validate_candidate_artifacts(candidate_model)

            frozen_eval_path = _resolve_path(paths_cfg.get("frozen_eval_data", {}).get("eval_subset"))
            if frozen_eval_path is None:
                raise ValueError("Config missing paths.frozen_eval_data.eval_subset")
            if not frozen_eval_path.is_file():
                raise FileNotFoundError(f"Frozen eval subset not found: {frozen_eval_path}")

            eval_wrapper_result = execute_evaluation(
                candidate_model=candidate_model,
                eval_data_path=frozen_eval_path,
                outputs_dir=outputs_dir,
                metric_name=metric_name,
            )

            candidate_metrics_path = Path(eval_wrapper_result["metrics_path"]).resolve()
            candidate_outputs_path = Path(eval_wrapper_result["outputs_path"]).resolve()

        if not eval_wrapper_result.get("ok", True):
            comparison_result = {
                "timestamp_utc": timestamp_now(),
                "classification": RUNTIME_OR_CONFIG_ERROR,
                "metric_name": metric_name,
                "baseline_score": None,
                "candidate_score": None,
                "absolute_min_threshold": None,
                "max_allowed_drop": None,
                "actual_drop": None,
                "sample_count_expected": None,
                "sample_count_candidate": 0,
                "artifacts_ok": False,
                "outputs_ok": False,
                "reasons": [
                    f"evaluation wrapper failed: {eval_wrapper_result.get('error_message', 'unknown error')}"
                ],
            }
        else:
            # Re-validate candidate artifacts before comparison.
            validate_required_files(
                base_path=Path("/"),
                filenames=[str(candidate_metrics_path), str(candidate_outputs_path)],
            )

            comparison_result = compare_rq4_results(
                baseline_metrics_path=baseline_metrics_path,
                candidate_metrics_path=candidate_metrics_path,
                baseline_outputs_path=baseline_outputs_path,
                candidate_outputs_path=candidate_outputs_path,
                thresholds_path=resolved["thresholds_config"],
            )

        classification = str(comparison_result.get("classification", RUNTIME_OR_CONFIG_ERROR))
        exit_code = map_classification_to_exit_code(classification)

        compare_manifest_path = manifests_dir / "rq4_comparison_result.json"
        write_json(compare_manifest_path, comparison_result)

        summary = {
            "timestamp_start_utc": start_ts,
            "timestamp_end_utc": timestamp_now(),
            "candidate_path": str((args.candidate_model or "(external artifacts)") if not args.skip_eval else "(artifact-only mode)"),
            "baseline_path": str(baseline_metrics_path.parent),
            "metric_used": comparison_result.get("metric_name"),
            "classification": classification,
            "reasons": comparison_result.get("reasons", []),
            "exit_code": exit_code,
            "config_path": str(config_path),
            "thresholds_path": str(resolved["thresholds_config"]),
            "baseline_metrics_path": str(baseline_metrics_path),
            "candidate_metrics_path": str(candidate_metrics_path),
            "baseline_outputs_path": str(baseline_outputs_path),
            "candidate_outputs_path": str(candidate_outputs_path),
            "evaluation_execution": eval_wrapper_result,
            "comparison_manifest_path": str(compare_manifest_path),
        }

        summary_path = manifests_dir / "rq4_gate_summary.json"
        write_json(summary_path, summary)

        print(f"[RQ4 GATE] classification={classification} exit_code={exit_code}")
        print(f"[RQ4 GATE] summary={summary_path}")
        return exit_code

    except Exception as exc:  # noqa: BLE001
        summary["timestamp_end_utc"] = timestamp_now()
        summary["classification"] = RUNTIME_OR_CONFIG_ERROR
        summary["reasons"] = [f"{type(exc).__name__}: {exc}"]
        summary["exit_code"] = EXIT_RUNTIME_ERROR

        manifests_dir = Path("rq4_evaluation/manifests")
        ensure_dir(manifests_dir)
        summary_path = manifests_dir / "rq4_gate_summary.json"
        write_json(summary_path, summary)

        print(f"[RQ4 GATE] runtime/config error: {exc}")
        print(f"[RQ4 GATE] summary={summary_path}")
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
