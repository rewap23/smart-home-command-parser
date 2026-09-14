.PHONY: install format lint test coverage run check

install:
	uv sync

format:
	uv run ruff format src tests

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

test:
	uv run pytest

coverage:
	uv run pytest --cov=smart_home_parser --cov-report=term-missing

run:
	uv run uvicorn smart_home_parser.api:app --reload --host 0.0.0.0 --port 8000

check: lint test

generate-data:
	uv run python scripts/generate_dataset.py

split-data:
	uv run python scripts/split_dataset.py

build-tokenizer:
	uv run python scripts/build_tokenizer.py

train-smoke:
	uv run python -m smart_home_parser.train \
		--epochs 1 \
		--batch-size 32 \
		--embedding-dim 32 \
		--num-heads 4 \
		--num-layers 1 \
		--feedforward-dim 64 \
		--artifact-dir artifacts/smoke \
		--device cpu

evaluate:
	uv run python -m smart_home_parser.evaluate \
		--test-path data/processed/test.jsonl \
		--artifact-dir artifacts/model-v0.1.0 \
		--device cpu

check:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts
	uv run pytest