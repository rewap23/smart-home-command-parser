from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def load_jsonl(path: Path) -> list[Record]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(records: list[Record], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/smart_home_commands.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    records = load_jsonl(args.input)

    fields = ("intent", "action", "device", "location", "value", "unit")
    split_names = ("train", "validation", "test")
    split_records: dict[str, list[Record]] = {
        split_name: [] for split_name in split_names
    }
    reserved_indexes: set[int] = set()

    # Reserve one record for every label value in every split. This is
    # especially important for numeric values, which are predicted as classes.
    for field in fields:
        values = {str(record["labels"][field]) for record in records}
        for value in sorted(values):
            for split_name in split_names:
                candidates = [
                    index
                    for index, record in enumerate(records)
                    if index not in reserved_indexes
                    and str(record["labels"][field]) == value
                ]
                if not candidates:
                    raise ValueError(
                        f"Not enough records to place {field}={value!r} "
                        f"in {split_name}."
                    )
                index = candidates[0]
                reserved_indexes.add(index)
                split_records[split_name].append(records[index])

    remaining_records = [
        record for index, record in enumerate(records) if index not in reserved_indexes
    ]
    random.shuffle(remaining_records)

    train_count = round(len(records) * 0.70) - len(split_records["train"])
    validation_count = round(len(records) * 0.15)
    split_records["train"].extend(remaining_records[:train_count])
    split_records["validation"].extend(
        remaining_records[train_count : train_count + validation_count]
    )
    split_records["test"].extend(remaining_records[train_count + validation_count :])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split in split_records.items():
        random.shuffle(split_records[split_name])

        output_path = args.output_dir / f"{split_name}.jsonl"
        write_jsonl(split, output_path)

    summary = {
        "seed": args.seed,
        "input_examples": len(records),
        "splits": {
            split_name: {
                "examples": len(split),
                "template_count": len({str(record["template_id"]) for record in split}),
                "intent_counts": dict(
                    sorted(
                        Counter(record["labels"]["intent"] for record in split).items()
                    )
                ),
                "action_counts": dict(
                    sorted(
                        Counter(record["labels"]["action"] for record in split).items()
                    )
                ),
            }
            for split_name, split in split_records.items()
        },
    }

    summary_path = args.output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote split files to: {args.output_dir}")
    print(f"Wrote split summary to: {summary_path}")


if __name__ == "__main__":
    main()
