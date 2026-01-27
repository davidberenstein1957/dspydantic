.PHONY: help install install-dev install-docs install-all test test-unit test-integration test-docs docs docs-serve docs-build clean lint format check pre-commit pre-commit-install pre-commit-run pre-commit-update setup ci watch watch-test watch-lint

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package and pre-commit hooks
	uv pip install -e .
	uv pip install -e ".[dev]" || true
	uv run pre-commit install || echo "Note: Pre-commit installation skipped. Run 'make install-dev' if you need pre-commit."

install-dev: ## Install the package with dev dependencies and pre-commit hooks
	uv pip install -e ".[dev]"
	uv run pre-commit install

install-docs: ## Install the package with docs dependencies
	uv pip install -e ".[docs]"

install-all: ## Install the package with all dependencies and pre-commit hooks
	uv pip install -e ".[dev,docs]"
	uv run pre-commit install

pre-commit-install: ## Install pre-commit hooks (requires dev dependencies)
	uv run pre-commit install

pre-commit-run: ## Run pre-commit on all files
	uv run pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	uv run pre-commit autoupdate

test: ## Run all tests
	uv run pytest

test-unit: ## Run unit tests only
	uv run pytest tests/unit/

test-integration: ## Run integration tests only
	uv run pytest tests/integration/

test-docs: ## Test documentation examples
	uv run pytest tests/test_docs.py

docs-build: ## Build documentation
	uv pip install -e ".[docs]"
	uv run mkdocs build --strict

docs-serve: ## Serve documentation locally with auto-reload (watches for changes)
	uv pip install -e ".[docs]"
	uv pip install -e ".[dev]" || true
	@echo "Starting mkdocs serve with file watching..."
	@uv run watchmedo shell-command --patterns="*.py" --recursive --command='uv run mkdocs build --strict' --wait src & \
	WATCH_PID=$$!; \
	trap 'kill $$WATCH_PID 2>/dev/null' EXIT INT TERM; \
	uv run mkdocs serve --dev-addr=127.0.0.1:8000; \
	kill $$WATCH_PID 2>/dev/null || true

docs: docs-build ## Alias for docs-build

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

lint: ## Run linters
	uv run ruff check src/ tests/
	uv run black --check src/ tests/
	uv run mypy src/

format: ## Format code automatically
	uv run black src/ tests/
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/
	@echo "Running pre-commit to ensure formatting..."
	@uv run pre-commit run --files src/**/*.py tests/**/*.py || true

check: lint test ## Run linting and tests

pre-commit: pre-commit-run ## Alias for pre-commit-run

setup: install-dev ## Complete setup: install dev deps (includes pre-commit hooks)
	@echo "Setup complete! Pre-commit hooks are now installed and will run on git commit."

ci: pre-commit-run lint test ## Run all CI checks (pre-commit, lint, test)

watch: watch-test ## Alias for watch-test
