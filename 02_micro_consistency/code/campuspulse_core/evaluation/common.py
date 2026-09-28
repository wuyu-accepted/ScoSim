from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STATE_FIELDS = (
    "attention", "concern", "trust", "satisfaction", "support", "opposition",
    "activity", "voice", "service_strain", "rumor_belief", "verified_belief",
    "evidence_acceptance",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def state_vector(record: dict[str, Any], phase: str = "post_state") -> np.ndarray:
    state = record.get(phase)
    if not isinstance(state, dict):
        raise ValueError(f"record lacks {phase}: agent={record.get('agent_id')} tick={record.get('tick')}")
    return np.asarray([float(state[field]) for field in STATE_FIELDS], dtype=np.float64)


def group_records(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]):
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
