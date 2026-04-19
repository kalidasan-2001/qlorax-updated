"""Small reusable helpers for the isolated RQ4 evaluation module."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

PathLike = str | Path


def _as_path(path: PathLike) -> Path:
    """Convert a string/path input into a ``Path`` object."""
    return path if isinstance(path, Path) else Path(path)


def ensure_dir(path: PathLike) -> Path:
    """Ensure a directory exists and return it as a ``Path``.

    Args:
        path: Directory path to create if missing.

    Returns:
        The resolved directory path.

    Raises:
        OSError: If the directory cannot be created.
    """
    dir_path = _as_path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_yaml(path: PathLike) -> Any:
    """Load and parse a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If YAML parsing fails.
    """
    file_path = _as_path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {file_path}: {exc}") from exc


def load_json(path: PathLike) -> Any:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON parsing fails.
    """
    file_path = _as_path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc


def write_json(path: PathLike, data: Any) -> None:
    """Write data to a JSON file using UTF-8 and stable formatting.

    Args:
        path: Output JSON file path.
        data: Serializable Python data.

    Raises:
        ValueError: If data cannot be serialized as JSON.
        OSError: If the file cannot be written.
    """
    file_path = _as_path(path)
    ensure_dir(file_path.parent)

    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except TypeError as exc:
        raise ValueError(f"Data is not JSON serializable for {file_path}: {exc}") from exc


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of JSON objects.

    Empty lines are ignored.

    Args:
        path: Path to JSONL file.

    Returns:
        A list of parsed JSON objects (one per non-empty line).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any line is invalid JSON or not an object.
    """
    file_path = _as_path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {file_path}:{line_no}: {exc}") from exc

            if not isinstance(parsed, dict):
                raise ValueError(
                    f"JSONL row must be an object at {file_path}:{line_no}; "
                    f"got {type(parsed).__name__}."
                )
            rows.append(parsed)

    return rows


def write_jsonl(path: PathLike, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows to a JSONL file (one JSON object per line).

    Args:
        path: Output JSONL file path.
        rows: Iterable of mapping-like row objects.

    Raises:
        ValueError: If any row cannot be serialized.
        OSError: If the file cannot be written.
    """
    file_path = _as_path(path)
    ensure_dir(file_path.parent)

    with file_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            try:
                f.write(json.dumps(dict(row), ensure_ascii=False))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid JSONL row at index {idx} for {file_path}: {exc}") from exc
            f.write("\n")


def timestamp_now() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def validate_required_files(base_path: PathLike, filenames: Sequence[str]) -> list[Path]:
    """Validate required files under ``base_path``.

    Args:
        base_path: Root directory containing required files.
        filenames: Relative file names expected under ``base_path``.

    Returns:
        List of resolved required file paths.

    Raises:
        FileNotFoundError: If one or more required files are missing.
    """
    root = _as_path(base_path)
    resolved_paths = [root / name for name in filenames]
    missing = [p for p in resolved_paths if not p.is_file()]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required files: {missing_list}")
    return resolved_paths


def safe_float(value: Any) -> float | None:
    """Safely convert a value to ``float``.

    Args:
        value: Any input value.

    Returns:
        Converted float value, or ``None`` if conversion fails.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    # Optional smoke check for quick local sanity validation.
    print("rq4_utils sanity check")
    print(f"timestamp_now: {timestamp_now()}")
    print(f"safe_float('0.82'): {safe_float('0.82')}")
    print(f"safe_float('not-a-number'): {safe_float('not-a-number')}")
