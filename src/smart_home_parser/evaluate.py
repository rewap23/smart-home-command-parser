from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from smart_home_parser.dataset import (
    LABEL_FIELDS,
    LabelEncoders,
    load_records,
)
from smart_home_parser.model import ModelConfig, SmartHomeTransformer
from smart_home_parser.tokenizer import WordTokenizer


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    return torch.device(requested_device)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def write_predictions(path: Path, predictions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction) + "\n")
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Smart Home Command Parser checkpoint "
            "on a held-out test split."
        )
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/processed/test.jsonl"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    test_path = args.test_path.resolve()
    output_path = (
        args.output_path.resolve()
        if args.output_path is not None
        else artifact_dir / "metrics.json"
    )
    predictions_path = output_path.with_name("predictions.jsonl")

    print(f"Working directory: {Path.cwd()}")
    print(f"Test split: {test_path}")
    print(f"Artifact directory: {artifact_dir}")
    print(f"Metrics output path: {output_path}")
    if args.save_predictions:
        print(f"Predictions output path: {predictions_path}")

    checkpoint_path = artifact_dir / "model.pt"
    vocab_path = artifact_dir / "vocab.json"
    encoder_path = artifact_dir / "label_encoders.json"

    for required_path in (
        test_path,
        checkpoint_path,
        vocab_path,
        encoder_path,
    ):
        require_file(required_path)

    device = resolve_device(args.device)
    print(f"Evaluation device: {device}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_checkpoint_keys = {
        "model_state_dict",
        "model_config",
        "epoch",
    }
    missing_keys = required_checkpoint_keys - set(checkpoint)
    if missing_keys:
        raise KeyError(f"Checkpoint is missing required keys: {sorted(missing_keys)}")

    config = ModelConfig(**checkpoint["model_config"])
    model = SmartHomeTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = WordTokenizer.load(vocab_path)
    label_encoders = LabelEncoders.load(encoder_path)

    test_records = load_records(test_path)
    if not test_records:
        raise ValueError(f"Test split is empty: {test_path}")

    encoded_inputs = [
        tokenizer.encode(str(record["command"])) for record in test_records
    ]
    input_ids = torch.stack([item[0] for item in encoded_inputs])
    attention_masks = torch.stack([item[1] for item in encoded_inputs])
    test_dataset = TensorDataset(input_ids, attention_masks)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    loss_examples = 0
    total_examples = len(test_records)
    correct_by_field = {field: 0 for field in LABEL_FIELDS}
    exact_matches = 0
    intent_labels = sorted(
        set(label_encoders.field_to_id["intent"])
        | {str(record["labels"]["intent"]) for record in test_records}
    )
    intent_confusion = {
        actual: {predicted: 0 for predicted in intent_labels}
        for actual in intent_labels
    }
    predictions: list[dict[str, object]] = []

    with torch.no_grad():
        for batch_start, (batch_input_ids, batch_attention_masks) in enumerate(
            test_loader
        ):
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_masks = batch_attention_masks.to(device)
            logits = model(
                batch_input_ids,
                batch_attention_masks,
            )
            batch_size = batch_input_ids.size(0)
            batch_records = test_records[
                batch_start * args.batch_size : batch_start * args.batch_size
                + batch_size
            ]

            batch_predictions = {
                field: logits[field].argmax(dim=1) for field in LABEL_FIELDS
            }
            exact_match_mask = torch.ones(
                batch_size,
                dtype=torch.bool,
                device=device,
            )

            decoded_predictions = {
                field: [
                    label_encoders.decode(field, label_id)
                    for label_id in batch_predictions[field].tolist()
                ]
                for field in LABEL_FIELDS
            }
            for index, record in enumerate(batch_records):
                true_labels = {
                    field: str(record["labels"][field]) for field in LABEL_FIELDS
                }
                predicted_labels = {
                    field: decoded_predictions[field][index] for field in LABEL_FIELDS
                }

                for field in LABEL_FIELDS:
                    matches = predicted_labels[field] == true_labels[field]
                    correct_by_field[field] += int(matches)
                    exact_match_mask[index] &= matches

                    label_id = label_encoders.field_to_id[field].get(true_labels[field])
                    if label_id is not None:
                        total_loss += criterion(
                            logits[field][index].unsqueeze(0),
                            torch.tensor([label_id], device=device),
                        ).item()
                        loss_examples += 1

                actual = true_labels["intent"]
                predicted = predicted_labels["intent"]
                intent_confusion[actual][predicted] += 1

                if args.save_predictions:
                    predictions.append(
                        {
                            "id": record.get("id"),
                            "command": record["command"],
                            "true_labels": true_labels,
                            "predicted_labels": predicted_labels,
                            "exact_match": true_labels == predicted_labels,
                        }
                    )

            exact_matches += exact_match_mask.sum().item()

    metrics = {
        "loss": total_loss / loss_examples if loss_examples else None,
        "loss_examples": loss_examples,
        **{
            f"{field}_accuracy": correct_by_field[field] / total_examples
            for field in LABEL_FIELDS
        },
        "exact_match_accuracy": exact_matches / total_examples,
        "intent_confusion_matrix": intent_confusion,
    }

    output: dict[str, object] = {
        "evaluation_timestamp_utc": datetime.now(UTC).isoformat(),
        "split": "test",
        "examples": len(test_dataset),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "model_name": "smart-home-command-parser-transformer",
        "label_fields": list(LABEL_FIELDS),
        **metrics,
    }

    atomic_write_json(output_path, output)

    if args.save_predictions:
        write_predictions(predictions_path, predictions)

    if not output_path.is_file():
        raise RuntimeError(
            f"Evaluation completed but metrics file was not created: {output_path}"
        )

    print("\nEvaluation metrics:")
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nSUCCESS: wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
