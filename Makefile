.PHONY: test lint format build

train:
	uv run python scripts/train_gan.py

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts

build:
	uv build

install-hooks:
	uv run pre-commit install
