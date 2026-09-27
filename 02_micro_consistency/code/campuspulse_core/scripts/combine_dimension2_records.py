from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.dimension2 import case_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    identifiers = set()
    for path in args.records:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = case_id(row)
            if identifier in identifiers:
                raise ValueError(f"duplicate case ID: {identifier}")
            identifiers.add(identifier)
            rows.append(row)
    if len(rows) != 2048 or len(identifiers) != 2048:
        raise ValueError("combined horizontal records must contain 2048 unique cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"record_count": len(rows), "unique_case_count": len(identifiers)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
