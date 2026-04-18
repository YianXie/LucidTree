# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LucidTree is a Go (board game) AI engine with a Django REST API. It combines a policy-value neural network with MCTS and minimax search to analyze board positions.

## Commands

All commands use `uv` as the package manager (install via `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
# Setup
uv sync --dev

# Run CLI engine demo
make lucidtree

# Run Django API server (port 9000)
make runserver

# Testing
make test                                                  # All tests
uv run pytest tests/test_board_rules.py -v                # Single file
uv run pytest tests/test_board_rules.py::test_function -v # Single test

# Code quality (all run in CI)
make format      # ruff format + isort
uv run ruff check .
uv run mypy . --show-error-codes
uv run pip-audit
uv run bandit -c pyproject.toml -r .

# Run all CI checks locally
make ci-local

# Migrations
python -m api.manage makemigrations
python -m api.manage migrate
```

## Architecture

### Repository Layout

```
src/
  lucidtree/      # Go AI engine (Python)
    cli/          # CLI entry point
    go/           # Board, Move, Player, Rules classes
    engine/       # Orchestrates analysis (selects algorithm)
    mcts/         # Monte Carlo Tree Search
    minimax/      # Alpha-beta pruning
    nn/           # Neural network model, training, inference
      model.py    # PolicyValueNetwork (CNN)
      agent.py    # Move selection from network output
      train.py    # Training loop
      datasets/   # SGF parsing and NPZ dataset loading
  api/            # Django REST API
    api/          # Django project settings + root URLs
    game_api/     # /api/analyze/ endpoint (views, serializers, services)
    common/       # Shared exceptions and utilities
tests/            # pytest suite
data/             # Raw SGF files and processed .npz shards
models/           # Pretrained checkpoints (checkpoint_19x19.pt)
scripts/          # CI helper and model download scripts
```

### Neural Network (`lucidtree.nn`)

**PolicyValueNetwork** is a 10-layer CNN with two heads:
- **Policy head**: probability distribution over 362 actions (19×19 board + pass)
- **Value head**: scalar win probability for the current position

Input is a 6-channel board encoding of the last 8 moves plus the color to play. Training uses supervised learning from ~400+ professional SGF games (combined cross-entropy + MSE loss, mixed precision on GPU).

### Search Algorithms (`lucidtree.mcts`, `lucidtree.minimax`)

- **MCTS**: PUCT-based tree search. Each `Node` stores visit counts (N), accumulated values (W), and policy priors (P). The network provides priors and leaf evaluations.
- **Minimax**: Shallow depth-limited alpha-beta pruning with heuristic evaluation.
- **NN-only**: Direct network inference with no search.

The `engine` module selects algorithm based on the API request parameters.

### REST API (`src/api/`)

Single endpoint: `POST /api/analyze/`
- **Request**: JSON with board state, move history, algorithm choice, and search parameters
- **Response**: Top N moves with policy probability, win rate, and visit counts
- **Flow**: `views.py` → `services.analyze()` → builds `Board` from move history → runs selected algorithm → serializes results

Development uses SQLite; production uses PostgreSQL via `DATABASE_URL` env var.

### Board Coordinate System

- **GTP notation** (external): `A1`–`T19` (column letter + row number from bottom)
- **Internal**: `(row, col)` tuples, 0-indexed from top-left
- **Pass move**: `(-1, -1)`

Key constants (in `src/lucidtree/constants.py`): `BOARD_SIZE=19`, `KOMI=7.5`, `NUM_SIMULATIONS=1000`, `CHANNEL_SIZE=6`

### Environment Variables

Copy `src/api/.env.example` to `src/api/.env`:
```
ENVIRONMENT=development   # or "production"
SECRET_KEY=<django-key>   # auto-provided in dev if omitted
DATABASE_URL=...          # defaults to SQLite in dev
```
