#!/usr/bin/env python3
"""
Compare two reproducibility run folders and fail on mismatches.

Compares:
- Config snapshots
- Saved metadata JSON files
- Artifact/model SHA256 hashes (from manifest if present; fallback direct hashing)
- Deterministic inference outputs (JSONL)

Outputs:
- Console summary
- JSON report

Exit codes:
- 0: runs match
- 2: mismatches found
- 1: usage/runtime error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_json(obj: Any) -> Any:
    """Stable normalization for deterministic comparison/reporting."""
    if isinstance(obj, dict):
        return {k: _normalize_json(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_normalize_json(v) for v in obj]
    return obj


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_existing(base: Path, candidates: List[str]) -> Optional[Path]:
    for rel in candidates:
        p = base / rel
        if p.exists() and p.is_file():
            return p
    return None


def _read_jsonl(path: Path) -> List[Any]:
    rows: List[Any] = []
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


def _diff_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Simple key-wise diff for report readability."""
    keys = sorted(set(a.keys()) | set(b.keys()))
    changed = {}
    for k in keys:
        av = a.get(k, "<missing>")
        bv = b.get(k, "<missing>")
        if av != bv:
            changed[k] = {"a": av, "b": bv}
    return changed


def _load_config(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
        return _normalize_json(yaml.safe_load(text))
    return text




def _normalize_metadata_for_compare(rel_path: str, obj: Any) -> Any:
    """Drop only expected per-run metadata fields that are not reproducibility signals."""
    if not isinstance(obj, dict):
        return obj

    normalized = dict(obj)

    if rel_path == "metadata/run_metadata.json":
        for key in ("run_name", "run_dir", "timestamp", "environment_metadata_path"):
            normalized.pop(key, None)

        # Keep train_metrics strict, but ignore timing/throughput noise.
        tm = normalized.get("train_metrics")
        if isinstance(tm, dict):
            for k in ("train_runtime", "train_samples_per_second", "train_steps_per_second"):
                tm.pop(k, None)

    elif rel_path == "metadata/environment.json":
        # Environment capture timestamp naturally differs between runs.
        normalized.pop("timestamp_utc", None)

    return normalized



def _normalize_artifact_hash_manifest_for_compare(hashes: Dict[str, str]) -> Dict[str, str]:
    """Ignore only known run-specific hash entries that are not reproducibility signals."""
    ignore_keys = {
        "metadata\\environment.json", "metadata/environment.json",  # timestamp_utc differs each run
        "metadata\\run_metadata.json", "metadata/run_metadata.json",  # run name/dir/timing fields differ
        "model\\training_args.bin", "model/training_args.bin",  # contains run-specific output/logging paths
    }
    return {k: v for k, v in hashes.items() if k not in ignore_keys}

def _collect_fallback_model_hashes(run_dir: Path) -> Dict[str, str]:
    model_dir = run_dir / "model"
    if not model_dir.exists():
        return {}

    hashes: Dict[str, str] = {}
    for p in sorted(model_dir.rglob("*")):
        if p.is_file():
            hashes[str(p.relative_to(run_dir))] = _sha256_file(p)
    return hashes


def compare_runs(run_a: Path, run_b: Path) -> Dict[str, Any]:
    mismatches: List[str] = []

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_a": str(run_a),
        "run_b": str(run_b),
        "comparisons": {},
        "mismatches": [],
        "status": "match",
    }

    # 1) Config comparison
    cfg_a = _first_existing(run_a, ["config/repro_config.used.yaml", "config/repro_config.yaml", "repro_config.yaml"])
    cfg_b = _first_existing(run_b, ["config/repro_config.used.yaml", "config/repro_config.yaml", "repro_config.yaml"])

    cfg_comp = {"found_a": str(cfg_a) if cfg_a else None, "found_b": str(cfg_b) if cfg_b else None}
    if cfg_a and cfg_b:
        try:
            a_obj = _load_config(cfg_a)
            b_obj = _load_config(cfg_b)
            equal = a_obj == b_obj
            cfg_comp["equal"] = equal
            if not equal:
                mismatches.append("config")
        except Exception as exc:
            cfg_comp["error"] = str(exc)
            mismatches.append("config_error")
    else:
        cfg_comp["equal"] = False
        cfg_comp["note"] = "Missing config in one or both runs"
        mismatches.append("config_missing")
    report["comparisons"]["config"] = cfg_comp

    # 2) Metadata comparison
    meta_files = ["metadata/run_metadata.json", "metadata/environment.json"]
    meta_comp: Dict[str, Any] = {}
    for rel in meta_files:
        pa, pb = run_a / rel, run_b / rel
        item = {"a_exists": pa.exists(), "b_exists": pb.exists()}
        if pa.exists() and pb.exists():
            try:
                a_obj = _normalize_metadata_for_compare(rel, _normalize_json(_load_json(pa)))
                b_obj = _normalize_metadata_for_compare(rel, _normalize_json(_load_json(pb)))
                eq = a_obj == b_obj
                item["equal"] = eq
                if not eq:
                    # Keep diff concise
                    if isinstance(a_obj, dict) and isinstance(b_obj, dict):
                        d = _diff_dict(a_obj, b_obj)
                        # Trim huge diffs
                        keys = list(d.keys())[:50]
                        item["diff_keys_sample"] = keys
                    mismatches.append(f"metadata:{rel}")
            except Exception as exc:
                item["equal"] = False
                item["error"] = str(exc)
                mismatches.append(f"metadata_error:{rel}")
        else:
            item["equal"] = False
            mismatches.append(f"metadata_missing:{rel}")
        meta_comp[rel] = item
    report["comparisons"]["metadata"] = meta_comp

    # 3) Hash comparison (manifest first, fallback to direct model hashing)
    hash_a_path = run_a / "metadata" / "artifact_hashes.json"
    hash_b_path = run_b / "metadata" / "artifact_hashes.json"
    hash_comp: Dict[str, Any] = {
        "hash_manifest_a": str(hash_a_path) if hash_a_path.exists() else None,
        "hash_manifest_b": str(hash_b_path) if hash_b_path.exists() else None,
    }

    if hash_a_path.exists() and hash_b_path.exists():
        try:
            ha = _normalize_artifact_hash_manifest_for_compare(_normalize_json(_load_json(hash_a_path)))
            hb = _normalize_artifact_hash_manifest_for_compare(_normalize_json(_load_json(hash_b_path)))
            eq = ha == hb
            hash_comp["equal"] = eq
            if not eq:
                if isinstance(ha, dict) and isinstance(hb, dict):
                    hash_comp["diff"] = _diff_dict(ha, hb)
                mismatches.append("artifact_hashes")
        except Exception as exc:
            hash_comp["equal"] = False
            hash_comp["error"] = str(exc)
            mismatches.append("artifact_hashes_error")
    else:
        # Fallback: compare hashes of all files under model/
        ha = _collect_fallback_model_hashes(run_a)
        hb = _collect_fallback_model_hashes(run_b)
        eq = ha == hb and bool(ha or hb)
        hash_comp["equal"] = eq
        hash_comp["fallback_model_hashing"] = True
        hash_comp["count_a"] = len(ha)
        hash_comp["count_b"] = len(hb)
        if not eq:
            hash_comp["diff"] = _diff_dict(ha, hb)
            mismatches.append("model_hash_fallback")
    report["comparisons"]["artifact_hashes"] = hash_comp

    # 4) Deterministic inference outputs
    inf_a = _first_existing(run_a, ["fixed_prompt_outputs.jsonl", "inference/fixed_prompt_outputs.jsonl", "outputs/fixed_prompt_outputs.jsonl"])
    inf_b = _first_existing(run_b, ["fixed_prompt_outputs.jsonl", "inference/fixed_prompt_outputs.jsonl", "outputs/fixed_prompt_outputs.jsonl"])

    inf_comp: Dict[str, Any] = {
        "found_a": str(inf_a) if inf_a else None,
        "found_b": str(inf_b) if inf_b else None,
    }
    if inf_a and inf_b:
        try:
            ra = _read_jsonl(inf_a)
            rb = _read_jsonl(inf_b)
            na = [_normalize_json(x) for x in ra]
            nb = [_normalize_json(x) for x in rb]
            eq = na == nb
            inf_comp["equal"] = eq
            inf_comp["rows_a"] = len(na)
            inf_comp["rows_b"] = len(nb)
            if not eq:
                # find first mismatch index
                first = None
                for i, (xa, xb) in enumerate(zip(na, nb)):
                    if xa != xb:
                        first = i
                        break
                if first is None and len(na) != len(nb):
                    first = min(len(na), len(nb))
                inf_comp["first_mismatch_index"] = first
                mismatches.append("inference_outputs")
        except Exception as exc:
            inf_comp["equal"] = False
            inf_comp["error"] = str(exc)
            mismatches.append("inference_outputs_error")
    else:
        inf_comp["equal"] = False
        inf_comp["note"] = "Missing deterministic inference output file in one or both runs"
        mismatches.append("inference_outputs_missing")

    report["comparisons"]["inference_outputs"] = inf_comp

    report["mismatches"] = mismatches
    report["status"] = "mismatch" if mismatches else "match"
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two reproducibility runs")
    p.add_argument("--run-a", required=True, help="Run folder name or absolute path for run A")
    p.add_argument("--run-b", required=True, help="Run folder name or absolute path for run B")
    p.add_argument(
        "--outputs-root",
        default="reproducibility/outputs",
        help="Root directory containing run folders (used when run-a/run-b are names)",
    )
    p.add_argument(
        "--report",
        default=None,
        help="Path to JSON report output (default: reproducibility/manifests/compare_<a>_vs_<b>.json)",
    )
    return p.parse_args()


def _resolve_run_path(value: str, outputs_root: Path) -> Path:
    p = Path(value)
    if p.exists():
        return p.resolve()
    alt = outputs_root / value
    return alt.resolve()


def _print_summary(report: Dict[str, Any]) -> None:
    print("=" * 72)
    print("Reproducibility Run Comparison")
    print("=" * 72)
    print(f"run_a: {report['run_a']}")
    print(f"run_b: {report['run_b']}")
    print(f"status: {report['status']}")

    comps = report.get("comparisons", {})
    print("\nChecks:")
    for k in ("config", "metadata", "artifact_hashes", "inference_outputs"):
        item = comps.get(k, {})
        if k == "metadata":
            # metadata has per-file map
            overall = all(v.get("equal", False) for v in item.values()) if item else False
            print(f"  - {k}: {'OK' if overall else 'MISMATCH'}")
        else:
            print(f"  - {k}: {'OK' if item.get('equal', False) else 'MISMATCH'}")

    mismatches = report.get("mismatches", [])
    if mismatches:
        print("\nMismatches found:")
        for m in mismatches:
            print(f"  - {m}")
    else:
        print("\nNo mismatches found. Runs are reproducibly equivalent for compared artifacts.")


def main() -> int:
    args = parse_args()

    outputs_root = Path(args.outputs_root)
    run_a = _resolve_run_path(args.run_a, outputs_root)
    run_b = _resolve_run_path(args.run_b, outputs_root)

    if not run_a.exists() or not run_a.is_dir():
        print(f"[ERROR] run-a folder not found: {run_a}", file=sys.stderr)
        return 1
    if not run_b.exists() or not run_b.is_dir():
        print(f"[ERROR] run-b folder not found: {run_b}", file=sys.stderr)
        return 1

    report = compare_runs(run_a, run_b)

    # Report path
    if args.report:
        report_path = Path(args.report)
    else:
        safe_a = run_a.name.replace("/", "_")
        safe_b = run_b.name.replace("/", "_")
        report_path = Path("reproducibility/manifests") / f"compare_{safe_a}_vs_{safe_b}.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_summary(report)
    print(f"\nJSON report: {report_path}")

    return 2 if report.get("status") == "mismatch" else 0


if __name__ == "__main__":
    raise SystemExit(main())
