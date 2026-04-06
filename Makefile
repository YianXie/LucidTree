.PHONY: help install test format ci-local lucidtree runserver clean

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install all dependencies"
	@echo "  test         - Run all tests"
	@echo "  format       - Format code"
	@echo "  ci-local     - Run all CI checks locally"
	@echo "  lucidtree    - Run the cli main.py file"
	@echo "  runserver    - Run the Django development server"
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

# Format code
format:
	@echo "Formatting code..."
	uv run ruff format . && uv run isort .

# Run all CI checks locally
ci-local:
	@./scripts/ci-local.sh

# Run the cli main.py file
lucidtree:
	@echo "Running the cli main.py file..."
	uv pip install -e .
	lucidtree

# Run the Django development server
runserver:
	@echo "Running the Django development server..."
	uv pip install -e .
	uv run python -m api.manage runserver 9000

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete