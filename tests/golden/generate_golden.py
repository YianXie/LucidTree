"""
Generate the MCTS golden master.

    uv run python tests/golden/generate_golden.py

Writes ``tests/golden/golden.npz`` containing ``root.N``, ``root.W`` and
``root.P`` for a fixed set of positions searched at a fixed simulation budget.
Any optimization to the search is required to reproduce these arrays exactly;
``tests/test_golden.py`` regenerates them and compares.

Determinism notes
-----------------
* ``torch.set_num_threads(1)``: thread count changes the reduction order inside
  the convolutions, which perturbs the low bits of the policy/value output and
  can flip near-ties in ``select_action``. The golden master and every
  verification run must be single-threaded.
* ``device=cpu``: MPS and CPU do not produce bit-identical convolution output.
* Dirichlet noise is disabled, so the search draws no random numbers at all.
* Positions are hardcoded move sequences, never sampled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from lucidtree.constants import BOARD_SIZE
from lucidtree.go.board import Board
from lucidtree.go.player import Player
from lucidtree.mcts.node import Node
from lucidtree.mcts.search import MCTS

GOLDEN_PATH = Path(__file__).with_name("golden.npz")

# ---- Search parameters under study (do not change without regenerating) ----
NUM_SIMULATIONS = 2048
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.0
DIRICHLET_EPSILON = 0.0
MAX_TIME_MS = None
MODEL = "latest"
# ----------------------------------------------------------------------------

PASS = "pass"

Moves = list[Any]

# A fixed 24-move opening. Every point is isolated enough that no move
# captures and none is suicide, so all prefixes are legal by construction.
OPENING: list[tuple[int, int]] = [
    (3, 3),
    (15, 15),
    (3, 15),
    (15, 3),
    (2, 5),
    (5, 16),
    (16, 5),
    (13, 2),
    (9, 9),
    (3, 9),
    (9, 3),
    (15, 9),
    (9, 15),
    (5, 2),
    (2, 13),
    (16, 13),
    (13, 16),
    (6, 6),
    (12, 12),
    (6, 12),
    (12, 6),
    (7, 3),
    (3, 7),
    (11, 15),
]

# Black plays (0, 1) and (1, 0), White is caught in the corner at (0, 0).
# Appended after an 8-move prefix so the capture happens in context.
CAPTURE: Moves = [*OPENING[:8], (0, 1), (0, 0), (1, 0)]

# Canonical ko shape around (9, 9). Black's last move captures the single
# white stone at (9, 9) and is itself a lone stone with one liberty, so the
# root position has White to play with the ko point set.
KO: Moves = [
    (8, 9),
    (9, 9),
    (10, 9),
    (8, 10),
    (9, 8),
    (10, 10),
    (3, 3),
    (9, 11),
    (9, 10),
]

POSITIONS: list[tuple[str, Moves]] = [
    *(
        (f"opening_{n:02d}", list(OPENING[:n]))
        for n in (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
    ),
    ("capture", CAPTURE),
    ("ko", KO),
    ("pass_early", [*OPENING[:6], PASS]),
    ("pass_late", [*OPENING[:12], PASS]),
]


def build_position(moves: Moves) -> tuple[Board, Player]:
    """
    Replay a fixed move sequence onto a fresh board.

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


def run_position(mcts: MCTS, moves: Moves, **overrides: Any) -> Node:
    """
    Search one fixed position with the study's control variables.

    Args:
        mcts (MCTS): a searcher already bound to the CPU model.
        moves (Moves): the move sequence defining the position.
        **overrides: extra kwargs forwarded to ``MCTS.run``.

    Returns:
        Node: the searched root node.
    """
    board, to_play = build_position(moves)
    kwargs: dict[str, Any] = {
        "num_simulations": NUM_SIMULATIONS,
        "c_puct": C_PUCT,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "dirichlet_epsilon": DIRICHLET_EPSILON,
        "max_time_ms": MAX_TIME_MS,
    }
    kwargs.update(overrides)
    return mcts.run(board=board, to_play=to_play, **kwargs)


def make_searcher(model: str = MODEL) -> MCTS:
    """
    Build the deterministic single-threaded CPU searcher.

    Args:
        model (str): the checkpoint name under ``models/``.

    Returns:
        MCTS: the searcher.
    """
    torch.set_num_threads(1)
    return MCTS(model=model, device=torch.device("cpu"))


def compute_golden(
    mcts: MCTS | None = None, verbose: bool = False, **overrides: Any
) -> dict[str, np.ndarray]:
    """
    Search every golden position and collect the root statistics.

    Args:
        mcts (MCTS | None): reuse an existing searcher, or build one.
        verbose (bool): print per-position progress.
        **overrides: extra kwargs forwarded to ``MCTS.run``.

    Returns:
        dict[str, np.ndarray]: ``"<name>__N"``, ``"<name>__W"``, ``"<name>__P"``
            arrays plus a ``"__names__"`` entry giving the position order.
    """
    if mcts is None:
        mcts = make_searcher()

    out: dict[str, np.ndarray] = {
        "__names__": np.array([name for name, _ in POSITIONS])
    }
    for name, moves in POSITIONS:
        root = run_position(mcts, moves, **overrides)
        out[f"{name}__N"] = root.N.copy()
        out[f"{name}__W"] = root.W.copy()
        out[f"{name}__P"] = root.P.copy()
        if verbose:
            print(f"  {name:<12} visits={int(root.N.sum()):>6}")
    return out


def main() -> None:
    """
    Regenerate ``tests/golden/golden.npz`` from scratch.
    """
    print(f"Generating golden master at N={NUM_SIMULATIONS}, c_puct={C_PUCT}")
    print(f"positions={len(POSITIONS)}, threads=1, device=cpu\n")
    # Typed as Any so mypy does not try to match the keys against
    # savez_compressed's keyword-only parameters.
    data: dict[str, Any] = dict(compute_golden(verbose=True))
    np.savez_compressed(GOLDEN_PATH, **data)
    print(f"\nWrote {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
