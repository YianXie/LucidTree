# fmt: off

import time
from typing import Any

import torch

from lucidtree.constants import BOARD_SIZE
from lucidtree.go.board import Board
from lucidtree.go.coordinates import row_col_to_gtp
from lucidtree.go.player import Player
from lucidtree.nn.agent import (load_model, pick_move_mcts, pick_move_minimax,
                                pick_move_nn)

# fmt: on

_NN_REQUIRED_BOARD_SIZE = BOARD_SIZE


def analyze_position(
    board: Board, to_play: Player, algo: str, params: dict[str, Any]
) -> dict[str, Any]:
    """
    Analyze a position

    Args:
        board (Board): the board
        to_play (Player): the player to play
        algo (str): the algorithm to use
        params (dict[str, Any]): the parameters for the algorithm

    Raises:
        ValueError: if algo is 'mcts' or 'nn' and board.size != 19

    Returns:
        dict[str, Any]: the analyzed position
    """
    if algo in ("mcts", "nn") and board.size != _NN_REQUIRED_BOARD_SIZE:
        raise ValueError(
            f"Algorithm '{algo}' only supports {_NN_REQUIRED_BOARD_SIZE}x{_NN_REQUIRED_BOARD_SIZE} boards, "
            f"but a {board.size}x{board.size} board was provided."
        )

    start = time.perf_counter()

    match algo:
        case "mcts":
            num_simulations = params.get("num_simulations", 1000)
            c_puct = params.get("c_puct", 1.5)

            best_move = pick_move_mcts(
                board, to_play, num_simulations=num_simulations, c_puct=c_puct
            )

            stats: dict[str, Any] = {
                "num_simulations": num_simulations,
                "c_puct": c_puct,
            }

        case "nn":
            model_name = params.get("model_name", None)

            model = load_model(model_name=model_name)
            best_move, _, _ = pick_move_nn(
                model,
                board,
                device=(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            )

            stats = {
                "model_name": model_name,
            }

        case "minimax":
            depth = params.get("depth", 3)

            best_move = pick_move_minimax(board, to_play, depth=depth)

            stats = {
                "depth": depth,
            }

        case _:
            raise ValueError(f"Invalid algorithm {algo}")

    end = time.perf_counter()
    elapsed_ms = round((end - start) * 1000, 2)

    return {
        "best_move": row_col_to_gtp(*best_move),
        "algorithm": algo,
        "stats": {
            **stats,
            "elapsed_ms": elapsed_ms,
        },
    }
