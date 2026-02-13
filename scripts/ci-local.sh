#!/bin/bash

# Local CI script to run the same checks as GitHub Actions
set -e

echo "🚀 Running local CI checks..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo -e "\n${YELLOW}Backend Checks${NC}"
echo "=================="

# Install dependencies
echo "Installing backend dependencies..."
uv sync --dev

# Run Ruff formatting check
echo "Checking code formatting with Ruff..."
if uv run ruff format --check .; then
    print_status "Ruff formatting check passed"
else
    print_error "Ruff formatting check failed. Run 'ruff format .' to fix."
    exit 1
fi

# Run Ruff formatting check
echo "Static type checking with mypy"
if uv run mypy --show-error-codes; then
    print_status "mypy type checking passed"
else
    print_error "mypy type checking failed."
    exit 1
fi

# Run isort import sorting check
echo "Checking import sorting with isort..."
if uv run isort --check-only --diff .; then
    print_status "isort import sorting check passed"
else
    print_error "isort import sorting check failed. Run 'isort .' to fix."
    exit 1
fi

# Run pip audit (security checks)
echo "Running pip audit..."
if uv run pip-audit; then
    print_status "pip audit passed"
else
    print_error "pip audit found issues."
    exit 1
fi

echo -e "\n${GREEN}🎉 All CI checks passed!${NC}"
echo "Your code is ready to be pushed to the repository."
