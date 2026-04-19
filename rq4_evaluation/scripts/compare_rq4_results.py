"""Compare RQ4 candidate evaluation artifacts against a blessed baseline.

This module is intentionally small and self-contained for thesis-focused,
controlled CI/CD gate experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    # Package-style import (preferred when run from repository root).
    from rq4_evaluation.scripts.rq4_utils import (  # type: ignore
        load_json,
        load_yaml,
        read_jsonl,
        safe_float,
        timestamp_now,
        write_json,
    )
except ModuleNotFoundError:
    # Script-local import fallback.
    from rq4_utils import load_json, load_yaml, read_jsonl, safe_float, timestamp_now, write_json


Classification = str
PASS: Classification = "pass"
DEGRADED_RELEASE: Classification = "degraded_release"
RUNTIME_OR_CONFIG_ERROR: Classification = "runtime_or_config_error"


def _extract_metric_score(metrics_obj: Any, metric_name: str) -> float | None:
    """Extract a metric value from common minimal JSON metric structures.

    Supported shapes:
    - {"metric_name": "macro_f1", "metric_value": 0.82}
    - {"macro_f1": 0.82}
    - {"metrics": {"macro_f1": 0.82}}
    """
    if not isinstance(metrics_obj, dict):
        return None

    direct_name = metrics_obj.get("metric_name")
    if isinstance(direct_name, str) and direct_name == metric_name:
        return safe_float(metrics_obj.get("metric_value"))

    if metric_name in metrics_obj:
        return safe_float(metrics_obj.get(metric_name))

    nested_metrics = metrics_obj.get("metrics")
    if isinstance(nested_metrics, dict) and metric_name in nested_metrics:
        return safe_float(nested_metrics.get(metric_name))

    return None


def _validate_candidate_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Validate minimal required structure of candidate output rows."""
    reasons: list[str] = []
    required_fields = ("id", "prompt", "prediction", "label", "valid")

    for idx, row in enumerate(rows, start=1):
        missing = [field for field in required_fields if field not in row]
        if missing:
            reasons.append(f"candidate_outputs row {idx} missing fields: {missing}")
            continue

        if not isinstance(row["prompt"], str):
            reasons.append(f"candidate_outputs row {idx} field 'prompt' must be string")
        if not isinstance(row["prediction"], str):
            reasons.append(f"candidate_outputs row {idx} field 'prediction' must be string")
        if not isinstance(row["valid"], bool):
            reasons.append(f"candidate_outputs row {idx} field 'valid' must be boolean")

    return reasons


def compare_rq4_results(
    baseline_metrics_path: Path,
    candidate_metrics_path: Path,
    baseline_outputs_path: Path,
    candidate_outputs_path: Path,
    thresholds_path: Path,
) -> dict[str, Any]:
    """Run explicit RQ4 gate comparison checks and return structured result.

    The function returns classification metadata and does not force process exit
    behavior, so orchestration scripts can decide final CI exit semantics.
    """
    reasons: list[str] = []
    degradation_reasons: list[str] = []

    artifacts_ok = True
    outputs_ok = True

    baseline_score: float | None = None
    candidate_score: float | None = None
    metric_name = ""
    min_threshold: float | None = None
    max_allowed_drop: float | None = None
    actual_drop: float | None = None
    sample_count_expected: int | None = None
    sample_count_candidate = 0

    # 1) Required artifact files exist (input contract).
    for required in (
        baseline_metrics_path,
        candidate_metrics_path,
        baseline_outputs_path,
        candidate_outputs_path,
        thresholds_path,
    ):
        if not required.is_file():
            artifacts_ok = False
            reasons.append(f"required file missing: {required}")

    # If core files are missing, return early with runtime/config classification.
    if not artifacts_ok:
        return {
            "timestamp_utc": timestamp_now(),
            "classification": RUNTIME_OR_CONFIG_ERROR,
            "metric_name": metric_name,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "absolute_min_threshold": min_threshold,
            "max_allowed_drop": max_allowed_drop,
            "actual_drop": actual_drop,
            "sample_count_expected": sample_count_expected,
            "sample_count_candidate": sample_count_candidate,
            "artifacts_ok": artifacts_ok,
            "outputs_ok": outputs_ok,
            "reasons": reasons,
        }

    # Load thresholds and policy fields.
    try:
        thresholds = load_yaml(thresholds_path)
        metric_gate = thresholds.get("metric_gate", {}) if isinstance(thresholds, dict) else {}
        validation = thresholds.get("validation", {}) if isinstance(thresholds, dict) else {}

        metric_name = str(metric_gate.get("primary_metric_name", ""))
        min_threshold = safe_float(metric_gate.get("minimum_acceptable_threshold"))
        max_allowed_drop = safe_float(metric_gate.get("maximum_allowed_regression_drop"))

        expected_raw = validation.get("expected_sample_count")
        sample_count_expected = int(expected_raw) if expected_raw is not None else None
    except Exception as exc:  # noqa: BLE001 - explicit runtime/config capture
        reasons.append(f"failed to load/parse thresholds YAML: {exc}")

    # Load metric files and validate numeric metric availability.
    try:
        baseline_metrics = load_json(baseline_metrics_path)
        candidate_metrics = load_json(candidate_metrics_path)

        if not metric_name and isinstance(baseline_metrics, dict):
            # Fallback for minimal configs where metric_name is omitted.
            if isinstance(baseline_metrics.get("metric_name"), str):
                metric_name = baseline_metrics["metric_name"]

        baseline_score = _extract_metric_score(baseline_metrics, metric_name)
        if baseline_score is None:
            reasons.append(
                f"baseline metric '{metric_name}' missing or non-numeric in {baseline_metrics_path}"
            )

        candidate_score = _extract_metric_score(candidate_metrics, metric_name)
        if candidate_score is None:
            reasons.append(
                f"candidate metric '{metric_name}' missing or non-numeric in {candidate_metrics_path}"
            )
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"failed to load/parse metrics JSON: {exc}")

    # Load outputs and run parse/structure checks.
    try:
        baseline_rows = read_jsonl(baseline_outputs_path)
        candidate_rows = read_jsonl(candidate_outputs_path)
        sample_count_candidate = len(candidate_rows)

        # 4) Expected sample count matches.
        if sample_count_expected is not None and sample_count_candidate != sample_count_expected:
            outputs_ok = False
            reasons.append(
                "candidate sample count mismatch: "
                f"expected={sample_count_expected}, got={sample_count_candidate}"
            )

        # Optional baseline/candidate count consistency check.
        if len(baseline_rows) != len(candidate_rows):
            outputs_ok = False
            reasons.append(
                "baseline/candidate output size mismatch: "
                f"baseline={len(baseline_rows)}, candidate={len(candidate_rows)}"
            )

        # 6) Candidate predictions structurally valid.
        structural_reasons = _validate_candidate_rows(candidate_rows)
        if structural_reasons:
            outputs_ok = False
            reasons.extend(structural_reasons)
    except Exception as exc:  # noqa: BLE001
        outputs_ok = False
        reasons.append(f"failed to read/validate outputs JSONL: {exc}")

    # 7) Candidate metric meets minimum threshold.
    if candidate_score is not None and min_threshold is not None and candidate_score < min_threshold:
        degradation_reasons.append(
            f"candidate metric below minimum threshold: {candidate_score:.6f} < {min_threshold:.6f}"
        )

    # 8) Candidate metric does not drop too far below baseline.
    if baseline_score is not None and candidate_score is not None:
        actual_drop = baseline_score - candidate_score
        if max_allowed_drop is not None and actual_drop > max_allowed_drop:
            degradation_reasons.append(
                "candidate regression exceeds allowed drop: "
                f"actual_drop={actual_drop:.6f}, allowed={max_allowed_drop:.6f}"
            )

    # Classification policy (explicit and thesis-defensible).
    if reasons:
        classification = RUNTIME_OR_CONFIG_ERROR
    elif degradation_reasons:
        classification = DEGRADED_RELEASE
    else:
        classification = PASS

    all_reasons = reasons + degradation_reasons

    return {
        "timestamp_utc": timestamp_now(),
        "classification": classification,
        "metric_name": metric_name,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "absolute_min_threshold": min_threshold,
        "max_allowed_drop": max_allowed_drop,
        "actual_drop": actual_drop,
        "sample_count_expected": sample_count_expected,
        "sample_count_candidate": sample_count_candidate,
        "artifacts_ok": artifacts_ok,
        "outputs_ok": outputs_ok,
        "reasons": all_reasons,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone comparator runs."""
    parser = argparse.ArgumentParser(description="Compare RQ4 candidate vs baseline results.")
    parser.add_argument("--baseline-metrics", required=True, type=Path)
    parser.add_argument("--candidate-metrics", required=True, type=Path)
    parser.add_argument("--baseline-outputs", required=True, type=Path)
    parser.add_argument("--candidate-outputs", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def main() -> int:
    """Run comparator as a standalone script and print/write result JSON."""
    args = _parse_args()

    result = compare_rq4_results(
        baseline_metrics_path=args.baseline_metrics,
        candidate_metrics_path=args.candidate_metrics,
        baseline_outputs_path=args.baseline_outputs,
        candidate_outputs_path=args.candidate_outputs,
        thresholds_path=args.thresholds,
    )

    if args.report is not None:
        write_json(args.report, result)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    # Intentionally not mapping classification -> process exit code here.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
