# RQ4 Evaluation Module (Isolated Thesis Experiment)

## 1) Purpose

This module provides a **controlled RQ4 experiment** to evaluate whether a CI/CD-style release gate can detect and block degraded LoRA candidate releases when compared to a fixed, accepted baseline release.

## 2) Scope

The module is intentionally self-contained under `rq4_evaluation/` and is designed for thesis evaluation workflows only. It focuses on:

- comparing candidate outputs/metrics against a fixed baseline,
- applying explicit threshold rules,
- generating pass/fail decisions and auditable manifests.

It does **not** modify or refactor the existing main project pipeline.

## 3) Thesis-safe boundary statement

**This module evaluates controlled candidate releases against a fixed blessed baseline and does not claim full reproducibility or universal release safety for the entire legacy repository.**

Accordingly, thesis conclusions should be interpreted as evidence of **controlled gate effectiveness** within this isolated setup.

## 4) Folder structure overview

```text
rq4_evaluation/
  README.md
  __init__.py
  configs/
    rq4_gate_ci.yaml
    rq4_baseline.yaml
    rq4_thresholds.yaml
  data/
    frozen_eval_subset.jsonl
    frozen_prompts.jsonl
    dataset_manifest.json
  baselines/
    reference_metrics.json
    reference_outputs.jsonl
    reference_release_manifest.json
  manifests/
    .gitkeep
  outputs/
    .gitkeep
  scripts/
    rq4_utils.py
    run_rq4_gate.py
    compare_rq4_results.py
    build_rq4_report.py
  tests/
    test_compare_rq4_results.py
    test_threshold_logic.py
```

## 5) Inputs

Typical inputs for the gate experiment:

- **Frozen evaluation data**: `data/frozen_eval_subset.jsonl`
- **Frozen prompts**: `data/frozen_prompts.jsonl`
- **Dataset manifest**: `data/dataset_manifest.json`
- **Gate config**: `configs/rq4_gate_ci.yaml`
- **Threshold policy**: `configs/rq4_thresholds.yaml`
- **Baseline descriptors**: files in `baselines/`
- **Candidate release artifacts**: provided externally (by path/reference)

## 6) Baseline concept

The baseline is a **fixed blessed release** accepted as the control reference for RQ4. It is represented by:

- `baselines/reference_release_manifest.json`
- `baselines/reference_metrics.json`
- `baselines/reference_outputs.jsonl`

Baseline content should remain stable during an experiment window to preserve comparability.

## 7) Candidate release concept

A candidate release is the **new LoRA artifact under evaluation** (for example, a newly trained or updated checkpoint). The candidate is assessed against the fixed baseline using the same frozen prompts/data and threshold policy.

## 8) Evaluation gate concept

The evaluation gate is a deterministic decision layer that:

1. loads baseline references,
2. loads candidate results,
3. compares metrics/outputs under configured thresholds,
4. emits a gate decision (`PASS` or `FAIL`) and machine-readable manifests.

## 9) Pass/fail policy

Pass/fail decisions are defined by `configs/rq4_thresholds.yaml`.

At minimum, policy is expected to include checks such as:

- maximum allowed degradation relative to baseline,
- minimum acceptable pass rate or quality threshold,
- behavior on missing or invalid artifacts (typically fail-closed).

A candidate **fails** if any blocking threshold is violated.

## 10) Exit codes

Recommended process-level exit code contract for gate runners:

- `0`: Gate PASS
- `1`: Gate FAIL (degradation or threshold violation)
- `2`: Execution/configuration error (missing files, invalid config, runtime issues)

## 11) Local usage

Example local run pattern (script names are scaffolded and thesis-facing):

```bash
python rq4_evaluation/scripts/run_rq4_gate.py \
  --config rq4_evaluation/configs/rq4_gate_ci.yaml

python rq4_evaluation/scripts/compare_rq4_results.py \
  --baseline rq4_evaluation/baselines/reference_release_manifest.json \
  --thresholds rq4_evaluation/configs/rq4_thresholds.yaml

python rq4_evaluation/scripts/build_rq4_report.py \
  --input-manifest rq4_evaluation/manifests/<gate_manifest>.json
```

## 12) CI usage

A dedicated workflow is provided at:

- `.github/workflows/rq4-eval-ci.yml`

Intended CI behavior:

- run the gate on candidate artifacts,
- publish gate manifests/reports as CI artifacts,
- fail the CI job when the gate returns non-pass status.

## 13) Generated outputs and manifests

Expected generated artifacts are written under:

- `rq4_evaluation/outputs/` (evaluation outputs)
- `rq4_evaluation/manifests/` (gate decisions, comparisons, summaries)

Typical manifest content should include:

- experiment timestamp and identifiers,
- baseline and candidate references,
- threshold values used,
- comparison results and mismatch details,
- final gate decision and exit status.
