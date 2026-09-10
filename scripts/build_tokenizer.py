from __future__ import annotations

import argparse
import json
from pathlib import Path

from smart_home_parser.tokenizer import WordTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/vocab.json"),
    )
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--min-frequency", type=int, default=1)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.train_path.open(encoding="utf-8")
        if line.strip()
    ]
    texts = [str(record["command"]) for record in records]

    tokenizer = WordTokenizer(max_length=args.max_length)
    tokenizer.fit(texts, min_frequency=args.min_frequency)
    tokenizer.save(args.output_path)

    print(
        f"Saved vocabulary with {tokenizer.vocab_size} tokens "
        f"to {args.output_path}"
    )


if __name__ == "__main__":
    main()