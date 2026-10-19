"""Ask Beam Free, through OpenCode, to label closed review decisions.

A training row is kept only when two separate passes return the same label.
The gold test file is never written into the training set.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schema import LABELS, RUBRIC  # noqa: E402

OPENCODE = Path.home() / ".opencode" / "bin" / "opencode"
MODEL = "opencode/beam-free"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))


def prompt_for(batch: list[dict]) -> str:
    items = "\n".join(
        f"{row['id']}: {row['state']}" for row in batch
    )
    return (
        "Label every item. Reply with a JSON array only, in the same order, "
        "and no markdown. Each element is "
        '{"id":"<id>","review":"approve"|"request_changes"|"escalate"}.\n\n'
        f"Rubric:\n{RUBRIC}\n\nItems:\n{items}"
    )


def extract_json(text: str):
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array in Beam output")
    return json.loads(text[start : end + 1])


def call_beam(prompt: str) -> list[dict]:
    binary = OPENCODE if OPENCODE.exists() else "opencode"
    result = subprocess.run(
        [str(binary), "run", "--model", MODEL, prompt],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(output[-2000:])
    parsed = extract_json(output)
    if not isinstance(parsed, list):
        raise ValueError("Beam JSON was not a list")
    return parsed


def label_pass(rows: list[dict], batch_size: int) -> dict[str, str]:
    found: dict[str, str] = {}
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        parsed = call_beam(prompt_for(batch))
        by_id = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            review = item.get("review")
            if item.get("id") in {row["id"] for row in batch} and review in LABELS:
                by_id[item["id"]] = review
        missing = [row["id"] for row in batch if row["id"] not in by_id]
        if missing:
            # One retry, still a single pass. A second pass is a separate call.
            parsed = call_beam(prompt_for(batch))
            for item in parsed:
                if isinstance(item, dict) and item.get("review") in LABELS:
                    by_id[item["id"]] = item["review"]
        for row in batch:
            if row["id"] not in by_id:
                raise RuntimeError(f"Beam omitted {row['id']}")
            found[row["id"]] = by_id[row["id"]]
        print(f"labeled {min(offset + batch_size, len(rows))}/{len(rows)}", flush=True)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, default=ROOT / "data" / "train_states.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "beam_train.jsonl")
    parser.add_argument("--dropped", type=Path, default=ROOT / "data" / "beam_dropped.jsonl")
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()

    rows = read_jsonl(args.states)
    first = label_pass(rows, args.batch_size)
    second = label_pass(rows, args.batch_size)

    kept = []
    dropped = []
    for row in rows:
        left, right = first[row["id"]], second[row["id"]]
        record = {
            "id": row["id"],
            "state": row["state"],
            "questions": __import__("schema").QUESTIONS,
            "pass_a": left,
            "pass_b": right,
        }
        if left == right:
            record["gold"] = {"review": left}
            record["source"] = "beam-agreement"
            kept.append(record)
        else:
            record["source"] = "beam-disagreement"
            dropped.append(record)

    write_jsonl(args.out, kept)
    write_jsonl(args.dropped, dropped)
    print(f"kept {len(kept)} dropped {len(dropped)}")


if __name__ == "__main__":
    main()
