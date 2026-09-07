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