#!/usr/bin/env python3
"""
Lightweight reproducibility utilities for the isolated reproducibility module.

This file intentionally avoids dependencies on the main project codebase.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _set_env_var(key: str, value: str) -> None:
    """Set an environment variable as string."""
    os.environ[key] = str(value)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def set_reproducible(
    seed: int = 42,
    deterministic: bool = True,
    cpu_only: bool = True,
    num_threads: int = 1,
) -> Dict[str, Any]:
    """
    Configure a reproducible execution environment.

    Parameters
    ----------
    seed:
        Global seed value for Python, NumPy, and PyTorch (if installed).
    deterministic:
        If True, enable deterministic backends where available.
    cpu_only:
        If True, request CPU-only execution by masking CUDA visibility.
    num_threads:
        If > 0, limit common thread pools and (if available) PyTorch threads.

    Returns
    -------
    dict
        Summary of applied settings and availability of optional libraries.
    """
    seed = _safe_int(seed, 42)
    num_threads = _safe_int(num_threads, 1)

    # 1) Core env vars for determinism and hashing stability
    _set_env_var("PYTHONHASHSEED", str(seed))
    _set_env_var("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # 2) CPU-only mode request
    if cpu_only:
        # Mask CUDA devices for most frameworks
        _set_env_var("CUDA_VISIBLE_DEVICES", "")

    # 3) Thread limits (only if requested)
    if num_threads and num_threads > 0:
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            _set_env_var(var, str(num_threads))

    # 4) Python RNG
    random.seed(seed)

    # 5) Optional NumPy RNG
    numpy_available = False
    numpy_error: Optional[str] = None
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
        numpy_available = True
    except Exception as exc:  # pragma: no cover
        numpy_error = str(exc)

    # 6) Optional PyTorch determinism
    torch_available = False
    torch_error: Optional[str] = None
    torch_details: Dict[str, Any] = {
        "deterministic_algorithms": None,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "torch_num_threads": None,
    }

    try:
        import torch  # type: ignore

        torch_available = True

        # Seed torch RNGs
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            except Exception:
                # Safe to ignore if CUDA APIs are unavailable in this build
                pass

        # Threading
        if num_threads and num_threads > 0:
            try:
                torch.set_num_threads(num_threads)
                if hasattr(torch, "set_num_interop_threads"):
                    torch.set_num_interop_threads(max(1, num_threads))
            except Exception:
                pass

        if deterministic:
            # Deterministic algorithms (when available)
            try:
                torch.use_deterministic_algorithms(True)
                torch_details["deterministic_algorithms"] = True
            except Exception:
                torch_details["deterministic_algorithms"] = False

            # cuDNN flags (safe on CPU builds too if attributes exist)
            try:
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    torch_details["cudnn_deterministic"] = True
                    torch_details["cudnn_benchmark"] = False
            except Exception:
                pass
        else:
            # If caller disables deterministic mode, leave backend defaults unchanged.
            torch_details["deterministic_algorithms"] = False

        # Report current torch thread count if available
        try:
            torch_details["torch_num_threads"] = int(torch.get_num_threads())
        except Exception:
            torch_details["torch_num_threads"] = None

    except Exception as exc:  # pragma: no cover
        torch_error = str(exc)

    return {
        "seed": seed,
        "deterministic": bool(deterministic),
        "cpu_only": bool(cpu_only),
        "num_threads": num_threads,
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "numpy": {
            "available": numpy_available,
            "error": numpy_error,
        },
        "torch": {
            "available": torch_available,
            "error": torch_error,
            **torch_details,
        },
    }


def _safe_run_command(cmd: list[str]) -> Optional[str]:
    """Run command and return stripped output or None on failure."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return out.strip() or None
    except Exception:
        return None


def collect_environment_metadata() -> Dict[str, Any]:
    """
    Collect lightweight environment metadata for reproducibility records.

    Returns
    -------
    dict
        JSON-serializable metadata payload.
    """
    metadata: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
        "git": {
            "commit": _safe_run_command(["git", "rev-parse", "HEAD"]),
            "branch": _safe_run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": _safe_run_command(["git", "status", "--porcelain"]) not in (None, ""),
        },
        "libraries": {},
    }

    # Optional library versions
    for lib in ("numpy", "torch", "transformers", "peft", "datasets"):
        try:
            mod = __import__(lib)
            metadata["libraries"][lib] = getattr(mod, "__version__", "unknown")
        except Exception:
            metadata["libraries"][lib] = None

    # Extra torch runtime info if installed
    try:
        import torch  # type: ignore

        metadata["torch_runtime"] = {
            "cuda_available": bool(torch.cuda.is_available()) if hasattr(torch, "cuda") else False,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
            "num_threads": int(torch.get_num_threads()) if hasattr(torch, "get_num_threads") else None,
        }
    except Exception:
        metadata["torch_runtime"] = None

    return metadata


def save_environment_metadata(output_path: str | Path) -> Path:
    """
    Collect and save environment metadata to JSON.

    Parameters
    ----------
    output_path:
        Destination file path.

    Returns
    -------
    pathlib.Path
        Resolved path to the written metadata file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_environment_metadata()
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out.resolve()
