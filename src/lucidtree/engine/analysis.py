# fmt: off

import random
import time
from typing import Any, cast

import numpy as np
import torch

from lucidtree.constants import BOARD_SIZE
from lucidtree.go.board import Board
from lucidtree.go.coordinates import (gtp_to_index, gtp_to_row_col,
                                      row_col_to_gtp)
from lucidtree.go.exceptions import BadRequestError
from lucidtree.go.player import Player
from lucidtree.nn.agent import (get_policy_value, load_model, pick_moves_mcts,
                                pick_moves_minimax, pick_moves_nn)
from lucidtree.nn.model import PolicyValueNetwork

# fmt: on


def analyze_position(
    algo: str,
    board: Board,
    komi: float,
    rules: str,
    to_play: Player,
    params: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    """
    Analyze a position

    Args:
        algo (str): the algorithm to use
        board (Board): the board
        komi (float): the komi value to apply when calculating scores
        rules (str): the rules (japanese or chinese) to apply when calculating scores
        to_play (Player): the player to play
        params (dict[str, Any]): the parameters for the algorithm
        output (dict[str, Any]): the output for the algorithm

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

    max_time_ms = params.get("max_time_ms")
    if max_time_ms is not None:
        if not isinstance(max_time_ms, (int, float)) or max_time_ms < 0:
            raise BadRequestError(
                "params.max_time_ms must be non-negative when provided; "
                "use a positive value for a time limit, or 0 / omit for no limit."
            )
    use_time_limit = (
        max_time_ms is not None
        and isinstance(max_time_ms, (int, float))
        and max_time_ms > 0
    )

    seed = params.get("seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    include_top_moves = output.get("include_top_moves", 5)
    if not isinstance(include_top_moves, int) or include_top_moves < 1:
        raise BadRequestError(
            "output.include_top_moves must be a positive integer, "
            f"but got {include_top_moves}."
        )

    start = time.perf_counter()

    policy_model: PolicyValueNetwork | None = None
    policy_device: torch.device | None = None

    match algo:
        case "mcts":
            model_name = params.get("model", "checkpoint_19x19")
            num_simulations = params.get("num_simulations", 1000)
            c_puct = params.get("c_puct", 1.5)
            dirichlet_alpha = params.get("dirichlet_alpha", 0.0)
            dirichlet_epsilon = params.get("dirichlet_epsilon", 0.0)
            value_weight = params.get("value_weight", 1.0)
            policy_weight = params.get("policy_weight", 1.0)
            select_by = params.get("select_by", "visit_count")
            include_visits = output.get("include_visits", True)

            mcts_stats: dict[str, Any] = {}
            mcts_kw: dict[str, Any] = {
                "model": model_name,
                "num_simulations": num_simulations,
                "c_puct": c_puct,
                "dirichlet_alpha": dirichlet_alpha,
                "dirichlet_epsilon": dirichlet_epsilon,
                "value_weight": value_weight,
                "policy_weight": policy_weight,
                "select_by": select_by,
                "komi": komi,
                "rules": rules,
                "stats_out": mcts_stats,
                "include_top_moves": include_top_moves,
                "include_visits": include_visits,
            }
            if use_time_limit:
                mcts_kw["max_time_ms"] = max_time_ms

            top_moves, visits = pick_moves_mcts(board, to_play, **mcts_kw)

            stats = {
                "model": str(model_name) if model_name is not None else None,
                "num_simulations": num_simulations,
                "c_puct": c_puct,
                "dirichlet_alpha": dirichlet_alpha,
                "dirichlet_epsilon": dirichlet_epsilon,
                "value_weight": value_weight,
                "policy_weight": policy_weight,
                "select_by": select_by,
                "include_visits": include_visits,
                "simulations_run": mcts_stats.get("simulations_run", num_simulations),
            }
            if use_time_limit:
                stats["max_time_ms"] = max_time_ms

        case "nn":
            model_name = params.get("model", "checkpoint_19x19")
            policy_softmax_temperature = params.get("temperature", 0.0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            nn_kw: dict[str, Any] = {
                "device": device,
                "temperature": policy_softmax_temperature,
                "include_top_moves": include_top_moves,
            }
            if use_time_limit:
                nn_kw["max_time_ms"] = max_time_ms

            checkpoint_model = load_model(
                model=model_name,
                device=device,
            )
            top_moves = pick_moves_nn(model=checkpoint_model, board=board, **nn_kw)

            policy_model = checkpoint_model
            policy_device = device

            stats = {
                "model": str(model_name) if model_name is not None else None,
                "policy_softmax_temperature": policy_softmax_temperature,
            }

        case "minimax":
            depth = params.get("depth", 3)
            use_alpha_beta = params.get("use_alpha_beta", True)

            minimax_stats: dict[str, Any] = {}
            mm_kw: dict[str, Any] = {
                "depth": depth,
                "use_alpha_beta": use_alpha_beta,
                "komi": komi,
                "rules": rules,
                "stats_out": minimax_stats,
                "include_top_moves": include_top_moves,
            }
            if use_time_limit:
                mm_kw["max_time_ms"] = max_time_ms

            top_moves = pick_moves_minimax(board, to_play, **mm_kw)

            stats = {
                "depth": depth,
                "use_alpha_beta": use_alpha_beta,
                "search_depth_reached": minimax_stats.get(
                    "search_depth_reached", depth
                ),
            }
            if use_time_limit:
                stats["max_time_ms"] = max_time_ms

        case _:
            raise BadRequestError(f"Invalid algorithm {algo}")

    end = time.perf_counter()
    elapsed_ms = round((end - start) * 1000, 2)

    top_moves_gtp: list[dict[str, Any]] = []
    for top_move_row_col in top_moves:
        top_moves_gtp.append({"move": row_col_to_gtp(*top_move_row_col)})

    include_policy = output.get("include_policy", False)
    include_winrate = output.get("include_winrate", False)
    if include_policy or include_winrate:
        infer_device: torch.device
        if policy_model is None:
            model_name = params.get("model", "checkpoint_19x19")
            infer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            policy_model = load_model(model=model_name, device=infer_device)
        else:
            infer_device = cast(torch.device, policy_device)
        policy_softmax_temperature = float(params.get("temperature", 0.0))
        policy, value = get_policy_value(
            policy_model,
            board,
            device=infer_device,
            temperature=policy_softmax_temperature,
        )
        if include_policy:
            policy = policy.tolist()  # type: ignore
            stats["policy"] = policy
            for top_move_gtp in top_moves_gtp:
                top_move_gtp["policy"] = policy[gtp_to_index(top_move_gtp["move"])]
        if include_winrate:
            stats["winrate"] = value
            for top_move_gtp in top_moves_gtp:
                board.place_move(
                    gtp_to_row_col(top_move_gtp["move"]), to_play.get_color()
                )
                _, value = get_policy_value(
                    policy_model,
                    board,
                    device=infer_device,
                    temperature=policy_softmax_temperature,
                )
                top_move_gtp["winrate"] = value
                board.undo()
        if include_visits and visits is not None:
            for i in range(len(top_moves_gtp)):
                top_moves_gtp[i]["visits"] = visits[i]

    return {
        "top_moves": top_moves_gtp,
        "algorithm": algo,
        "stats": {
            **stats,
            "elapsed_ms": elapsed_ms,
        },
    }
