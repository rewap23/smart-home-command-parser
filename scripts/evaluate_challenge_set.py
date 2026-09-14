from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from smart_home_parser.dataset import LABEL_FIELDS, load_records


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def check_challenge_set_is_held_out(
    challenge_records: list[dict[str, Any]],
    split_paths: list[Path],
) -> None:
    challenge_ids = {
        str(record["id"])
        for record in challenge_records
        if record.get("id") is not None
    }
    challenge_commands = {
        str(record["command"]).strip().lower() for record in challenge_records
    }

    if not challenge_ids and not challenge_commands:
        raise ValueError("Challenge set has neither usable IDs nor commands.")

    for split_path in split_paths:
        if not split_path.is_file():
            continue

        split_records = load_records(split_path)
        split_ids = {
            str(record["id"])
            for record in split_records
            if record.get("id") is not None
        }
        split_commands = {
            str(record["command"]).strip().lower() for record in split_records
        }

        overlapping_ids = sorted(challenge_ids & split_ids)
        overlapping_commands = sorted(challenge_commands & split_commands)

        if overlapping_ids or overlapping_commands:
            details: list[str] = []

            if overlapping_ids:
                details.append(f"overlapping IDs: {overlapping_ids[:5]}")

            if overlapping_commands:
                details.append(f"overlapping commands: {overlapping_commands[:3]}")

            raise ValueError(
                f"Challenge-set leakage detected in {split_path}. " + "; ".join(details)
            )


def validate_record_schema(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("Challenge set is empty.")

    for index, record in enumerate(records, start=1):
        if "command" not in record:
            raise ValueError(f"Challenge record {index} has no 'command' field.")

        labels = record.get("labels")
        if not isinstance(labels, dict):
            raise ValueError(f"Challenge record {index} has no valid 'labels' object.")

        missing_fields = [field for field in LABEL_FIELDS if field not in labels]
        if missing_fields:
            raise ValueError(
                f"Challenge record {index} is missing label fields: {missing_fields}"
            )


def load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected prediction file was not created: {path}")

    return read_jsonl(path)


def calculate_challenge_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("Prediction output is empty.")

    total = len(predictions)
    exact_matches = sum(bool(record["exact_match"]) for record in predictions)

    correct_by_field = {field: 0 for field in LABEL_FIELDS}

    unsupported_total = 0
    unsupported_true_positive = 0
    unsupported_predicted_total = 0

    failures: list[dict[str, Any]] = []

    for record in predictions:
        true_labels = record["true_labels"]
        predicted_labels = record["predicted_labels"]

        for field in LABEL_FIELDS:
            if predicted_labels[field] == true_labels[field]:
                correct_by_field[field] += 1

        is_unsupported = true_labels["intent"] == "unsupported"
        predicted_unsupported = predicted_labels["intent"] == "unsupported"

        if is_unsupported:
            unsupported_total += 1

        if predicted_unsupported:
            unsupported_predicted_total += 1

        if is_unsupported and predicted_unsupported:
            unsupported_true_positive += 1

        if not record["exact_match"]:
            incorrect_fields = [
                field
                for field in LABEL_FIELDS
                if true_labels[field] != predicted_labels[field]
            ]

            failures.append(
                {
                    **record,
                    "incorrect_fields": incorrect_fields,
                }
            )

    unsupported_recall = (
        unsupported_true_positive / unsupported_total if unsupported_total else None
    )

    unsupported_precision = (
        unsupported_true_positive / unsupported_predicted_total
        if unsupported_predicted_total
        else None
    )

    return {
        "examples": total,
        "exact_match_accuracy": exact_matches / total,
        "field_accuracy": {
            field: correct_by_field[field] / total for field in LABEL_FIELDS
        },
        "unsupported_examples": unsupported_total,
        "unsupported_true_positives": unsupported_true_positive,
        "unsupported_predicted_examples": unsupported_predicted_total,
        "unsupported_recall": unsupported_recall,
        "unsupported_precision": unsupported_precision,
        "failures": failures,
    }


def format_percentage(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def build_report(
    metrics: dict[str, Any],
    predictions_path: Path,
) -> str:
    failures: list[dict[str, Any]] = metrics["failures"]

    lines = [
        "# Challenge Set Results",
        "",
        "## Summary",
        "",
        f"- Examples: {metrics['examples']}",
        (
            "- Exact-match accuracy: "
            f"{format_percentage(metrics['exact_match_accuracy'])}"
        ),
        (
            "- Unsupported-command recall: "
            f"{format_percentage(metrics['unsupported_recall'])}"
        ),
        (
            "- Unsupported-command precision: "
            f"{format_percentage(metrics['unsupported_precision'])}"
        ),
        f"- Prediction records: `{predictions_path}`",
        "",
        "## Per-field accuracy",
        "",
        "| Field | Accuracy |",
        "|---|---:|",
    ]

    for field in LABEL_FIELDS:
        lines.append(
            f"| {field} | {format_percentage(metrics['field_accuracy'][field])} |"
        )

    lines.extend(
        [
            "",
            "## Error analysis",
            "",
            f"- Exact-match failures: {len(failures)}",
            "",
        ]
    )

    if not failures:
        lines.extend(
            [
                "No challenge-set failures were observed.",
                "",
                (
                    "This still does not prove real-world robustness. "
                    "Expand the challenge set with new paraphrases, "
                    "speech-to-text errors, ambiguity, typos, novel device "
                    "aliases, and unsupported requests."
                ),
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| ID | Command | Incorrect fields | Expected intent | Predicted intent |",
            "|---|---|---|---|---|",
        ]
    )

    for failure in failures:
        command = str(failure["command"]).replace("|", "\\|")
        incorrect_fields = ", ".join(failure["incorrect_fields"])
        expected_intent = failure["true_labels"]["intent"]
        predicted_intent = failure["predicted_labels"]["intent"]

        lines.append(
            f"| {failure.get('id', 'N/A')} | {command} | "
            f"{incorrect_fields} | {expected_intent} | "
            f"{predicted_intent} |"
        )

    lines.extend(
        [
            "",
            "## Detailed failures",
            "",
        ]
    )

    for failure in failures:
        lines.extend(
            [
                f"### {failure.get('id', 'Unknown ID')}",
                "",
                f"**Command:** `{failure['command']}`",
                "",
                "**Expected parse:**",
                "",
                "```json",
                json.dumps(failure["true_labels"], indent=2),
                "```",
                "",
                "**Predicted parse:**",
                "",
                "```json",
                json.dumps(failure["predicted_labels"], indent=2),
                "```",
                "",
                ("**Incorrect fields:** " + ", ".join(failure["incorrect_fields"])),
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Smart Home Command Parser model on a "
            "manually authored challenge set."
        )
    )
    parser.add_argument(
        "--challenge-path",
        type=Path,
        default=Path("data/evaluation/challenge_commands.jsonl"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/model-v0.1.0"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/model-v0.1.0/challenge"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train.jsonl"),
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=Path("data/processed/validation.jsonl"),
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/processed/test.jsonl"),
    )
    args = parser.parse_args()

    challenge_path = args.challenge_path.resolve()
    artifact_dir = args.artifact_dir.resolve()
    output_dir = args.output_dir.resolve()

    require_file(challenge_path, "Challenge set")
    require_file(artifact_dir / "model.pt", "Model checkpoint")
    require_file(artifact_dir / "vocab.json", "Tokenizer vocabulary")
    require_file(
        artifact_dir / "label_encoders.json",
        "Label encoders",
    )

    challenge_records = read_jsonl(challenge_path)
    validate_record_schema(challenge_records)

    check_challenge_set_is_held_out(
        challenge_records=challenge_records,
        split_paths=[
            args.train_path.resolve(),
            args.validation_path.resolve(),
            args.test_path.resolve(),
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.jsonl"
    report_path = output_dir / "challenge_report.md"

    command = [
        sys.executable,
        "-m",
        "smart_home_parser.evaluate",
        "--test-path",
        str(challenge_path),
        "--artifact-dir",
        str(artifact_dir),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--output-path",
        str(metrics_path),
        "--save-predictions",
    ]

    print("Running evaluator:")
    print(" ".join(command))
    subprocess.run(command, check=True)

    predictions = load_predictions(predictions_path)
    if len(predictions) != len(challenge_records):
        raise RuntimeError(
            "Prediction count does not match challenge-set size: "
            f"{len(predictions)} predictions for "
            f"{len(challenge_records)} challenge records."
        )

    challenge_metrics = calculate_challenge_metrics(predictions)

    evaluation_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    evaluation_metrics["challenge_set"] = {
        "source_path": str(challenge_path),
        "examples": challenge_metrics["examples"],
        "unsupported_examples": challenge_metrics["unsupported_examples"],
        "unsupported_true_positives": challenge_metrics["unsupported_true_positives"],
        "unsupported_predicted_examples": challenge_metrics[
            "unsupported_predicted_examples"
        ],
        "unsupported_recall": challenge_metrics["unsupported_recall"],
        "unsupported_precision": challenge_metrics["unsupported_precision"],
        "exact_match_failures": len(challenge_metrics["failures"]),
    }
    evaluation_metrics["split"] = "challenge"

    metrics_path.write_text(
        json.dumps(evaluation_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = build_report(
        metrics=challenge_metrics,
        predictions_path=predictions_path,
    )
    write_text(report_path, report)

    print("\nChallenge-set summary:")
    print(
        f"Examples: {challenge_metrics['examples']}\n"
        "Exact-match accuracy: "
        f"{format_percentage(challenge_metrics['exact_match_accuracy'])}\n"
        "Unsupported recall: "
        f"{format_percentage(challenge_metrics['unsupported_recall'])}\n"
        "Unsupported precision: "
        f"{format_percentage(challenge_metrics['unsupported_precision'])}\n"
        f"Exact-match failures: {len(challenge_metrics['failures'])}"
    )
    print(f"\nSUCCESS: wrote challenge metrics to {metrics_path}")
    print(f"SUCCESS: wrote predictions to {predictions_path}")
    print(f"SUCCESS: wrote report to {report_path}")


if __name__ == "__main__":
    main()
