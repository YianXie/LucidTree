"""
Multi-process MCTS collection harness for the scaling study.

    # collect 40 positions at N=4096 across 4 workers
    uv run python -m scripts.collect_positions --positions 40 \
        --num-simulations 4096 --workers 4 --out data/collected

    # measure aggregate throughput at several worker counts
    uv run python -m scripts.collect_positions --bench 1,2,4,6,8

The engine itself stays strictly single-threaded and single-process; all
parallelism lives here. Each worker sets ``torch.set_num_threads(1)`` and loads
its own copy of the model, so no tensor is ever shared across processes and the
per-worker search is bit-identical to a serial run.

Results are written one file per position (``NNNN_name.npz``), so a crash or a
kill loses only the position that was in flight. Re-running the same command
skips positions whose output file already exists unless ``--overwrite`` is
given.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from lucidtree.constants import BOARD_SIZE
from lucidtree.go.board import Board
from lucidtree.go.player import Player

PASS = "pass"
Moves = list[Any]

# 121 pairwise non-adjacent points: three interleaved lattices of stride 3.
# Stones on these points never capture and never commit suicide, so every
# prefix of this sequence is a legal position by construction.
SPREAD_POINTS: list[tuple[int, int]] = [
    (row, col)
    for offset in (0, 1, 2)
    for row in range(offset, BOARD_SIZE, 3)
    for col in range(offset, BOARD_SIZE, 3)
]

_WORKER: dict[str, Any] = {}


def default_positions(count: int, stride: int = 3) -> list[tuple[str, Moves]]:
    """
    Build a deterministic ladder of positions of increasing stone count.

    Args:
        count (int): how many positions to produce.
        stride (int): additional stones per step.

    Returns:
        list[tuple[str, Moves]]: (name, move sequence) pairs.
    """
    positions = []
    for i in range(count):
        length = min(i * stride, len(SPREAD_POINTS))
        positions.append((f"spread_{length:03d}", list(SPREAD_POINTS[:length])))
    return positions


def load_positions(path: Path) -> list[tuple[str, Moves]]:
    """
    Read positions from JSON.

    The file is either a list of move lists, or a list of
    ``{"name": ..., "moves": [...]}`` objects. A move is ``[row, col]`` or the
    string ``"pass"``.

    Args:
        path (Path): the JSON file.

    Returns:
        list[tuple[str, Moves]]: (name, move sequence) pairs.
    """
    raw = json.loads(path.read_text())
    positions: list[tuple[str, Moves]] = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            name = str(item.get("name", f"pos_{i:04d}"))
            moves = item["moves"]
        else:
            name, moves = f"pos_{i:04d}", item
        parsed: Moves = [PASS if m == PASS else (int(m[0]), int(m[1])) for m in moves]
        positions.append((name, parsed))
    return positions


def build_position(moves: Moves) -> tuple[Board, Player]:
    """
    Replay a move sequence onto a fresh board.

    Args:
        moves (Moves): (row, col) tuples, or the string ``"pass"``.

    Returns:
        tuple[Board, Player]: the board and the player to play.
    """
    black = Player.black()
    white = Player.white()
    black.opponent = white
    white.opponent = black

    board = Board(BOARD_SIZE, black, white)
    for move in moves:
        if move == PASS:
            board.pass_move()
        else:
            board.place_move(move, board.get_current_player().get_color())
    return board, board.get_current_player()


def _init_worker(config: dict[str, Any]) -> None:
    """
    Per-process setup: pin to one thread, then load this worker's own model.

    Args:
        config (dict[str, Any]): the run configuration.
    """
    import torch

    from lucidtree.mcts.search import MCTS

    # Critical: without this every worker spawns its own intra-op thread pool,
    # the processes fight over the same cores, and aggregate throughput drops
    # below the single-worker number.
    torch.set_num_threads(1)

    _WORKER["config"] = config
    _WORKER["mcts"] = MCTS(model=config["model"], device=torch.device("cpu"))


def _run_one(task: tuple[int, str, Moves]) -> dict[str, Any]:
    """
    Search one position and write its result file.

    Args:
        task (tuple[int, str, Moves]): index, name and move sequence.

    Returns:
        dict[str, Any]: a summary record for the caller.
    """
    index, name, moves = task
    config = _WORKER["config"]
    mcts = _WORKER["mcts"]

    out_path = Path(config["out"]) / f"{index:04d}_{name}.npz"
    if out_path.exists() and not config["overwrite"]:
        return {"index": index, "name": name, "skipped": True, "seconds": 0.0}

    board, to_play = build_position(moves)

    start = time.perf_counter()
    root = mcts.run(
        board=board,
        to_play=to_play,
        num_simulations=config["num_simulations"],
        c_puct=config["c_puct"],
        dirichlet_alpha=config["dirichlet_alpha"],
        dirichlet_epsilon=config["dirichlet_epsilon"],
        **config["extra"],
    )
    elapsed = time.perf_counter() - start

    payload: dict[str, Any] = {
        "N": root.N,
        "W": root.W,
        "P": root.P,
        "legal_mask": root.legal_mask,
        "meta": np.array(
            json.dumps(
                {
                    "name": name,
                    "index": index,
                    "moves": [
                        m if m == PASS else [int(m[0]), int(m[1])] for m in moves
                    ],
                    "num_simulations": config["num_simulations"],
                    "simulations_run": mcts.simulations_run,
                    "c_puct": config["c_puct"],
                    "model": config["model"],
                    "seconds": elapsed,
                    "pid": os.getpid(),
                }
            )
        ),
    }
    snapshots = getattr(root, "snapshots", None)
    if snapshots:
        for budget, visits in snapshots.items():
            payload[f"snapshot_{int(budget)}"] = visits

    # Write-then-rename so a kill mid-write never leaves a half-file behind.
    # The temp name has to keep the .npz suffix, or savez_compressed appends
    # its own and the rename target would not exist.
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(out_path)

    return {
        "index": index,
        "name": name,
        "skipped": False,
        "seconds": elapsed,
        "simulations": int(mcts.simulations_run),
    }


def collect(
    positions: list[tuple[str, Moves]], config: dict[str, Any], workers: int
) -> dict[str, float]:
    """
    Run every position across a process pool.

    Args:
        positions (list[tuple[str, Moves]]): the work list.
        config (dict[str, Any]): the run configuration.
        workers (int): the number of worker processes.

    Returns:
        dict[str, float]: wall-clock seconds and aggregate sims/sec.
    """
    Path(config["out"]).mkdir(parents=True, exist_ok=True)
    tasks = [(i, name, moves) for i, (name, moves) in enumerate(positions)]

    start = time.perf_counter()
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers, initializer=_init_worker, initargs=(config,)
    ) as pool:
        done = 0
        simulations = 0
        for record in pool.imap_unordered(_run_one, tasks):
            done += 1
            simulations += int(record.get("simulations", 0))
            state = "skip" if record["skipped"] else f"{record['seconds']:6.2f}s"
            print(
                f"  [{done:>4}/{len(tasks)}] {record['name']:<14} {state}",
                flush=True,
            )
    wall = time.perf_counter() - start

    return {
        "wall_seconds": wall,
        "simulations": float(simulations),
        "sims_per_sec": simulations / wall if wall > 0 else 0.0,
    }


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build the worker configuration dict from parsed arguments."""
    extra: dict[str, Any] = {}
    if args.snapshots:
        extra["snapshot_powers_of_two"] = True
    return {
        "model": args.model,
        "num_simulations": args.num_simulations,
        "c_puct": args.c_puct,
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_epsilon": args.dirichlet_epsilon,
        "out": str(args.out),
        "overwrite": args.overwrite,
        "extra": extra,
    }


def main() -> None:
    """Parse arguments and either collect or benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=20)
    parser.add_argument("--positions-json", type=Path, default=None)
    parser.add_argument("--num-simulations", type=int, default=2048)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.0)
    parser.add_argument("--model", default="latest")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("data/collected"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--snapshots",
        action="store_true",
        help="also record root.N at every power of two",
    )
    parser.add_argument(
        "--bench",
        default=None,
        help="comma-separated worker counts to benchmark, e.g. 1,2,4,6,8",
    )
    args = parser.parse_args()

    # Belt and braces: the children re-import torch under the spawn start
    # method, and these are read at import time.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    if args.positions_json is not None:
        positions = load_positions(args.positions_json)
    else:
        positions = default_positions(args.positions)

    config = _config_from_args(args)

    if args.bench:
        counts = [int(c) for c in args.bench.split(",")]
        print(
            f"Benchmarking {len(positions)} positions x "
            f"N={args.num_simulations} at {counts} workers\n"
        )
        results = {}
        for workers in counts:
            config["out"] = str(Path(args.out) / f"bench_w{workers}")
            config["overwrite"] = True
            results[workers] = collect(positions, config, workers)
            print(
                f"  -> {workers} worker(s): "
                f"{results[workers]['sims_per_sec']:8.1f} sims/sec "
                f"({results[workers]['wall_seconds']:.1f}s wall)\n",
                flush=True,
            )

        baseline = results[counts[0]]["sims_per_sec"]
        print(f"{'workers':>8}  {'sims/sec':>10}  {'wall (s)':>9}  {'speedup':>8}")
        for workers in counts:
            row = results[workers]
            print(
                f"{workers:>8}  {row['sims_per_sec']:>10.1f}  "
                f"{row['wall_seconds']:>9.1f}  "
                f"{row['sims_per_sec'] / baseline:>7.2f}x"
            )
        return

    print(
        f"Collecting {len(positions)} positions at N={args.num_simulations} "
        f"across {args.workers} worker(s) into {args.out}\n"
    )
    summary = collect(positions, config, args.workers)
    print(
        f"\nDone in {summary['wall_seconds']:.1f}s "
        f"({summary['sims_per_sec']:.1f} sims/sec aggregate)"
    )


if __name__ == "__main__":
    main()
