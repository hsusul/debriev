PYTHON ?= python3

.PHONY: install dev migrate test

install:
	$(PYTHON) -m pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload

migrate:
	alembic upgrade head

test:
	pytest

