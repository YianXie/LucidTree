# fmt: off

import time
from typing import Any

import torch

from mini_katago.go.board import Board
from mini_katago.go.player import Player
from mini_katago.nn.agent import (load_model, pick_move_mcts,
                                  pick_move_minimax, pick_move_nn)

# fmt: on


def analyze_position(
    board: Board, to_play: Player, algo: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Analyze a position

    Args:
        board (Board): the board
        to_play (Player): the player to play
        algo (str): the algorithm to use
        params (dict[str, Any]): the parameters for the algorithm

    Returns:
        dict[str, Any]: the analyzed position
    """
    start = time.perf_counter()

    match algo:
        case "mcts":
            num_simulations = params.get("num_simulations", 1000)
            c_puct = params.get("c_puct", 1.5)

            best_move = pick_move_mcts(
                board, to_play, num_simulations=num_simulations, c_puct=c_puct
            )

            stats = {
                "best_move": best_move,
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
                "best_move": best_move,
                "model_name": model_name,
            }

        case "minimax":
            depth = params.get("depth", 3)

            best_move = pick_move_minimax(board, to_play, depth=depth)

            stats = {
                "best_move": best_move,
                "depth": depth,
            }

        case _:
            raise ValueError(f"Invalid algorithm {algo}")

    end = time.perf_counter()
    elapsed_ms = round((end - start) * 1000, 2)

    return {
        "best_move": str(best_move),
        "algorithm": algo,
        "stats": {
            **stats,
            "elapsed_ms": elapsed_ms,
        },
    }
