# fmt: off

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lucidtree.constants import BOARD_SIZE
from lucidtree.go.board import Board
from lucidtree.go.coordinates import row_col_to_gtp
from lucidtree.go.exceptions import BadRequestError
from lucidtree.go.player import Player
from lucidtree.nn.agent import (load_model, pick_move_mcts, pick_move_minimax,
                                pick_move_nn)

# fmt: on


def analyze_position(
    board: Board,
    to_play: Player,
    algo: str,
    params: dict[str, Any],
    model: Path | str | None = None,
) -> dict[str, Any]:
    """
    Analyze a position

    Args:
        board (Board): the board
        to_play (Player): the player to play
        algo (str): the algorithm to use
        params (dict[str, Any]): the parameters for the algorithm
        model (Path | str | None, optional): the path to the model to load. Defaults to None.
            If None, loads the default model from the models directory (checkpoint_19x19.pt).
            If a string, it is assumed to be the name of the model and is loaded from the models directory.
            If a Path, it is assumed to be the path to the model and is loaded from the given path.

    Raises:
        ValueError: if algo is 'mcts' or 'nn' and board.size != 19

    Returns:
        dict[str, Any]: the analyzed position
    """
    if board.size != BOARD_SIZE:
        raise BadRequestError(
            f"Algorithm '{algo}' only supports {BOARD_SIZE}x{BOARD_SIZE} boards, "
            f"but a {board.size}x{board.size} board was provided."
        )

    seed = params.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    start = time.perf_counter()

    match algo:
        case "mcts":
            model_name = params.get("model", model)
            num_simulations = params.get("num_simulations", 1000)
            c_puct = params.get("c_puct", 1.5)
            dirichlet_alpha = params.get("dirichlet_alpha", 0.0)
            dirichlet_epsilon = params.get("dirichlet_epsilon", 0.0)
            value_weight = params.get("value_weight", 1.0)
            policy_weight = params.get("policy_weight", 1.0)
            select_by = params.get("select_by", "visit_count")

            best_move = pick_move_mcts(
                board,
                to_play,
                model=model_name,
                num_simulations=num_simulations,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                value_weight=value_weight,
                policy_weight=policy_weight,
                select_by=select_by,
            )

            stats = {
                "model": str(model_name) if model_name is not None else None,
                "num_simulations": num_simulations,
                "c_puct": c_puct,
                "dirichlet_alpha": dirichlet_alpha,
                "dirichlet_epsilon": dirichlet_epsilon,
                "value_weight": value_weight,
                "policy_weight": policy_weight,
                "select_by": select_by,
            }

        case "nn":
            model_name = params.get("model", model)
            policy_softmax_temperature = params.get(
                "policy_softmax_temperature",
                params.get("temperature", 0.0),
            )
            use_value_head = params.get("use_value_head", True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint_model = load_model(
                model=model_name,
                device=device,
            )
            best_move, probability, value = pick_move_nn(
                checkpoint_model,
                board,
                device=device,
                temperature=policy_softmax_temperature,
            )

            stats = {
                "model": str(model_name) if model_name is not None else None,
                "policy_softmax_temperature": policy_softmax_temperature,
                "selected_move_probability": probability,
                "use_value_head": use_value_head,
            }
            if use_value_head:
                stats["value"] = value

        case "minimax":
            depth = params.get("depth", 3)
            use_alpha_beta = params.get("use_alpha_beta", True)

            best_move = pick_move_minimax(
                board,
                to_play,
                depth=depth,
                use_alpha_beta=use_alpha_beta,
            )

            stats = {
                "depth": depth,
                "use_alpha_beta": use_alpha_beta,
            }

        case _:
            raise BadRequestError(f"Invalid algorithm {algo}")

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
