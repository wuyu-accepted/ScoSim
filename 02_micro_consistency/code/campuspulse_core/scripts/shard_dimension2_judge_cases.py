from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministically shard unfinished D2 judge cases.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--exclude-case-id", action="append", default=[])
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("shards must be positive")
    cases = read_jsonl(args.cases)
    completed = {str(row["case_id"]) for row in read_jsonl(args.completed)}
    excluded = set(args.exclude_case_id)
    pending = [
        row for row in cases
        if str(row["case_id"]) not in completed and str(row["case_id"]) not in excluded
    ]
    buckets = [[] for _ in range(args.shards)]
    for row in sorted(pending, key=lambda value: str(value["case_id"])):
        index = int(hashlib.sha256(str(row["case_id"]).encode("utf-8")).hexdigest(), 16) % args.shards
        buckets[index].append(row)
    args.output_root.mkdir(parents=True, exist_ok=False)
    counts = []
    for index, rows in enumerate(buckets):
        write_jsonl(args.output_root / f"shard-{index:02d}.jsonl", rows)
        counts.append(len(rows))
    manifest = {
        "schema_version": "dimension2-judge-shards-v1",
        "total_case_count": len(cases),
        "completed_case_count": len(completed),
        "pending_case_count": len(pending),
        "excluded_case_ids": sorted(excluded),
        "shard_count": args.shards,
        "shard_counts": counts,
        "assignment": "sha256(case_id) modulo shard_count",
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
