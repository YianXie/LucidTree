# Makefile for LucidTree project

.PHONY: help install test lint format security ci-local all clean

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install all dependencies"
	@echo "  test         - Run all tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code"
	@echo "  security     - Run security checks"
	@echo "  ci-local     - Run all CI checks locally"
	@echo "  run          - Run the cli main.py file"
	@echo "  all 		  - Run all checks (CI & Tests)"
	@echo "  clean        - Clean up generated files"

# Install dependencies
install:
	@echo "Installing dependencies..."
	uv sync --dev

# Run tests
test:
	@echo "Running tests..."
	uv pip install -e .
	uv run pytest

# Run linting
lint:
	@echo "Running linting..."
	uv run ruff check .

# Format code
format:
	@echo "Formatting code..."
	uv run ruff format . && uv run isort .

# Run security checks
security:
	@echo "Running backend security checks..."
	uv run pip-audit

# Run all CI checks locally
ci-local:
	@./scripts/ci-local.sh

# Run the cli main.py file
run:
	@echo "Running the cli main.py file..."
	uv pip install -e .
	lucidtree

all:
	@echo "Running all checks..."
	@./scripts/ci-local.sh
	uv run pytest

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete