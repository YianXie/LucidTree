# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LucidTree is a Go (board game) AI engine with a Django REST API. It combines a policy-value neural network with MCTS and minimax search to analyze board positions.

## Commands

All commands use `uv` as the package manager (install via `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
# Setup
uv sync --dev

# Run Django API server (port 9000)
make runserver

# Testing
make test                                                  # All tests
uv run pytest tests/test_board_rules.py -v                # Single file
uv run pytest tests/test_board_rules.py::test_function -v # Single test

# Code quality (all run in CI)
make format      # ruff format + isort
uv run ruff check .
uv run isort --check-only --diff .
uv run mypy . --show-error-codes
uv run pip-audit
uv run bandit -c pyproject.toml -r .

# Run all CI checks locally
make ci-local

# Migrations
uv run python -m api.manage makemigrations
uv run python -m api.manage migrate

# Download the pretrained checkpoint into models/ (models/ is gitignored)
uv run python -m scripts.download_checkpoint

# MCTS throughput benchmark (CPU, single-threaded)
uv run python bench_throughput.py            # sweeps N = 256/1024/4096/16384
uv run python bench_throughput.py 16384      # single budget

# Multi-process position collection for the scaling study
uv run python -m scripts.collect_positions --positions 40 --workers 6
uv run python -m scripts.collect_positions --bench 1,2,4,6,8

# Regenerate the MCTS golden master (only when the search is meant to change)
uv run python tests/golden/generate_golden.py
```

**The MCTS search is under a behavior-preserving constraint.** `tests/test_golden.py`
pins `root.N`/`root.W`/`root.P` for 20 fixed positions at `N=2048`; `root.N` must match
*exactly*. Run it after any change under `mcts/`, `go/board.py` or `nn/features.py`. It
takes ~90s and skips when `models/latest.pt` is absent. Do not regenerate `golden.npz` to
make a failure go away — a failure means the search changed.

`make lucidtree` still exists and the `lucidtree` console script is declared in
`pyproject.toml`, but `src/lucidtree/cli/main.py` is currently empty (its contents were
removed in commit `5094447`), so the command does nothing. There is no working CLI demo —
exercise the engine through the API or directly from Python.

## Architecture

### Repository Layout

```
src/
  lucidtree/      # Go AI engine (Python)
    cli/          # Console-script entry point (main.py currently empty)
    common/       # paths.py (project-root lookup), logging.py
    go/           # Board, Move, Player, Rules, Game, coordinates, exceptions
                  # interactive_board.py = pygame board for local play
    engine/       # analysis.py (algorithm dispatch), winrate.py (per-move winrate)
    mcts/         # Monte Carlo Tree Search (search.py, node.py)
    minimax/      # Alpha-beta pruning
    nn/           # Neural network model, training, inference
      model.py       # PolicyValueNetwork (CNN)
      agent.py       # load_model + pick_moves_{mcts,nn,minimax}, get_policy_value
      features.py    # encode_board (6 planes), value_to_winrate
      train.py       # Training loop
      evaluate.py    # Validation/test evaluation
      split.py       # Split a game into training positions
      datasets/      # SGF parsing/tools, gokifu downloader, SGF + precomputed NPZ datasets
  api/            # Django REST API
    api/          # Django project settings + root URLs
    game_api/     # /api/ endpoints (views, serializers, services)
    common/       # Shared exceptions and request-parsing utilities
tests/            # pytest suite
  golden/         # MCTS golden master: generate_golden.py + golden.npz
scripts/          # ci-local.sh, download_checkpoint.py, collect_positions.py
data/             # Raw SGF files and processed .npz shards (gitignored)
models/           # Checkpoints, latest.pt (gitignored; fetched by download_checkpoint.py)
figures/          # Training plots
logs/             # dataset.log, training.log (gitignored)
render-build.sh   # Render deploy build: collectstatic, migrate, download checkpoint
```

### Neural Network (`lucidtree.nn`)

**PolicyValueNetwork** is a 10-layer CNN trunk (128 channels, Conv → GroupNorm → ReLU) with two heads:
- **Policy head**: 1×1 conv down to a 19×19 board map plus a separate pass logit from
  global-average-pooled trunk features → 362 logits
- **Value head**: 1×1 conv → linear stack → `tanh`, i.e. a scalar in `[-1, 1]` for the side to move
  (converted to a winrate percentage by `features.value_to_winrate`)

`features.encode_board` produces the 6 input planes: black stones, white stones, empty points,
side-to-play (all-ones plane when black), last move, ko point. It encodes only the current
position — there is no move history stack.

Training (`train.py`) is supervised on professional SGF games (`data/raw/sgf`, ~23k games,
preprocessed into ~300 `.npz` shards under `data/processed/{train,val,test}`), using
cross-entropy on policy (with label smoothing) + MSE on value, with AMP autocast/GradScaler
enabled on CUDA.

### Search Algorithms (`lucidtree.mcts`, `lucidtree.minimax`)

- **MCTS**: PUCT-based tree search. Each `Node` stores visit counts (N), accumulated values (W), and policy priors (P). The network provides priors and leaf evaluations.
- **Minimax**: Shallow depth-limited alpha-beta pruning with heuristic evaluation.
- **NN-only**: Direct network inference with no search.

MCTS specifics worth knowing before touching it:

- `Node.Q` returns `0.0` for unvisited actions. This deviates from AlphaZero (which uses
  FPU reduction) and is **deliberate** — it is a property of the system under study. Do
  not "fix" it.
- `Node.select_action` is NumPy-vectorized via `Q_all()`/`U_all()`. The scalar `Q()`/`U()`
  are kept for the public API. The vectorized versions reproduce the scalar float
  promotions exactly (float64 for `W/N`, a float32 rounding of `policy_weight * c_puct`
  before it meets the float32 prior). Changing those casts changes search results.
- `Board.copy()` — not `copy.deepcopy` — is used to fork boards per simulation. It copies
  the grid, both players (capture counts included), the ko point and the history list.
- `MCTS.run` kwargs beyond the obvious: `use_nn_cache` (default `True`) is a per-search
  transposition cache keyed on the encoded input tensor; hit rates are ~20% on an empty
  board and 1-3% once stones are down. `snapshot_powers_of_two` (default `False`) records
  `root.N` at 1, 2, 4, ... into `root.snapshots`, so one run to N_max yields every smaller
  budget. `MCTS.cache_stats` holds the last run's hit counts.
- Virtual loss, batched leaf evaluation and tree reuse are all deliberately absent: each
  changes which node gets selected, i.e. changes the algorithm.

`engine/analysis.py::analyze_position` dispatches on the request's `algo` (`"mcts"`, `"nn"`,
`"minimax"`), and handles `params` such as `seed` and `max_time_ms`. All algorithms require a
19×19 board; anything else raises `BadRequestError`.

### REST API (`src/api/`)

Routes are mounted under `/api/` (`src/api/game_api/urls.py`):

- `GET /api/health/` — liveness check, returns `{"status": "ok", "service": "lucidtree-api"}`
- `POST /api/analyze/` — request: `moves` (list of `[color, point]`), `to_play` (`"B"`/`"W"`),
  `algo`, `params`, `output`, optional `komi` (serializer default **6.5**, unlike
  `constants.KOMI = 7.5`) and `rules` (`"japanese"`/`"chinese"`).
  Response: `top_moves`, `algorithm`, `stats`.
- `POST /api/winrate/` — request: `moves`, optional `params` (`device`, `temperature`).
  Response: `winrate`, a per-move list of `{"black": …, "white": …}` percentages.

All three views are unauthenticated (`authentication_classes`/`permission_classes` empty).
Flow: `views.py` → `services.analyze()` / `services.winrate()` → builds a `Board` from the move
list → runs the selected algorithm → serializes results. `BadRequestError` maps to HTTP 400;
anything else is logged and returns HTTP 500.

Django settings use SQLite unconditionally (`src/api/api/settings.py`) — `dj-database-url` and
`psycopg2` are installed dependencies but nothing reads `DATABASE_URL` yet, so pointing the app
at PostgreSQL requires wiring it up first. Production serving uses gunicorn + whitenoise
(see `render-build.sh`); CORS/ALLOWED_HOSTS are hardcoded for localhost,
`lucidtree.onrender.com`, and `api.lucidtree.org`.

### Board Coordinate System

- **GTP notation** (external): `A1`–`T19` (column letter + row number from bottom, no `I`)
- **Internal**: `(row, col)` tuples, 0-indexed from top-left
- **Pass move**: `(-1, -1)` (`PASS_MOVE_POSITION`); pass index for the policy vector is `361` (`PASS_INDEX`)

Conversions live in `src/lucidtree/go/coordinates.py` (`gtp_to_row_col`, `row_col_to_gtp`,
`gtp_to_index`, `index_to_gtp`, `row_col_to_index`, `index_to_row_col`).

Key constants (in `src/lucidtree/constants.py`): `BOARD_SIZE=19`, `KOMI=7.5`, `RULES="japanese"`,
`CHANNEL_SIZE=6`, `NUM_SIMULATIONS=1000`, `EXPLORATION_CONSTANT=1.5`, `MAX_GAME_DEPTH=50`,
`SHARD_SIZE=50_000`, plus colors `BLACK_COLOR=1` / `WHITE_COLOR=-1` / `EMPTY_COLOR=0`.

### Environment Variables

Copy `src/api/.env.example` to `src/api/.env`:
```
ENVIRONMENT=""     # "development" for local dev; anything else (default) is treated as production
SECRET_KEY=""      # required unless ENVIRONMENT=development, which falls back to an insecure dev key
```
`ENVIRONMENT=development` also turns on `DEBUG`. The CI workflows additionally set
`DATABASE_URL`, but the settings module currently ignores it.
