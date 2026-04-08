# fmt: off

import time
from pathlib import Path
from typing import Any

import torch

from lucidtree.common.paths import get_project_root
from lucidtree.constants import (KOMI, PASS_INDEX, PASS_MOVE_POSITION, RULES,
                                 WHITE_COLOR)
from lucidtree.go.board import Board
from lucidtree.go.coordinates import index_to_row_col
from lucidtree.go.move import Move
from lucidtree.go.player import Player
from lucidtree.minimax.search import next_best_move
from lucidtree.nn.features import encode_board
from lucidtree.nn.model import PolicyValueNetwork

# fmt: on

root = get_project_root()


@torch.no_grad()
def load_model(
    model: Path | str | None = None,
    device: torch.device | None = None,
) -> PolicyValueNetwork:
    """
    Load the Neural Network model

    Args:
        model (Path | str | None, optional): the path to the model to load. Defaults to None.
            If None, loads the default model from the models directory (checkpoint_19x19.pt).
            If a string, it is assumed to be the name of the model and is loaded from the models directory.
            If a Path, it is assumed to be the path to the model and is loaded from the given path.
        device (torch.device | None, optional): the device to load the model onto.
            If None, loads to CPU. Use CUDA when available for GPU inference.

    Raises:
        FileNotFoundError: if the model file does not exist

    Returns:
        PolicyValueNetwork: the loaded model
    """
    if isinstance(model, Path):
        path = model
    elif isinstance(model, str):
        path = root / "models" / f"{model}.pt"
    else:
        path = root / "models/checkpoint_19x19.pt"

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: '{path.name}'. "
            "Ensure the model is present in the models/ directory."
        )

    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    checkpoint_model = PolicyValueNetwork()
    checkpoint_model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint_model = checkpoint_model.to(device)
    return checkpoint_model


@torch.no_grad()
def pick_move_mcts(
    board: Board, to_play: Player, model: Path | str | None = None, **kwargs: Any
) -> tuple[int, int]:
    """
    Pick the next best move given the board, model, device, and temperature using the MCTS algorithm

    Args:
        board (Board): the current board state
        to_play (Player): the player to play
        model (Path | str | None, optional): the path to the model to load. Defaults to None.
            If None, loads the default model from the models directory (checkpoint_19x19.pt).
            If a string, it is assumed to be the name of the model and is loaded from the models directory.
            If a Path, it is assumed to be the path to the model and is loaded from the given path.
        **kwargs: additional keyword arguments

    Returns:
        tuple[int, int]: the move
    """
    from lucidtree.mcts.search import MCTS

    kw = dict(kwargs)
    stats_out = kw.pop("stats_out", None)

    mcts = MCTS(model=model, **kw)
    root = mcts.run(board=board, to_play=to_play, **kw)

    if stats_out is not None:
        stats_out["simulations_run"] = mcts.simulations_run

    pos = MCTS.pick_best_move_position(
        root, select_by=kwargs.get("select_by", "visit_count")
    )
    return pos


@torch.no_grad()
def pick_move_nn(
    model: PolicyValueNetwork,
    board: Board,
    device: torch.device,
    temperature: float = 0.0,
) -> tuple[tuple[int, int], float, float]:
    """
    Pick the next best move given the board, model, device, and temperature

    Args:
        model (PolicyValueNetwork): the model to use
        board (Board): the current board state
        device (torch.device): the PyTorch device
        temperature (float, optional): the temperature. Defaults to 0.0.

    Returns:
        tuple[tuple[int, int], float, float]: the move, probability, and value
    """
    model.eval()
    model = model.to(device)

    x = encode_board(board)
    x = x.unsqueeze(0).to(device)
    x = x.float()

    policy_logits, value = model(x)
    logits = policy_logits[0]
    value = value.item()

    if temperature <= 0.0:
        probs = torch.softmax(logits, dim=0)
    else:
        probs = torch.softmax(logits / temperature, dim=0)

    order = torch.argsort(probs, descending=True).tolist()

    for idx in order:
        if idx == PASS_INDEX:
            return PASS_MOVE_POSITION, probs[idx].item(), value

        else:
            move_pos = index_to_row_col(idx)
            if (
                board.move_is_valid(
                    Move(
                        move_pos[0], move_pos[1], board.get_current_player().get_color()
                    )
                )
                and board.get_move_at_position(move_pos).is_empty()
            ):
                return move_pos, probs[idx].item(), value

    return PASS_MOVE_POSITION, probs[idx].item(), value


def get_policy_value(
    model: PolicyValueNetwork,
    board: Board,
) -> tuple[torch.Tensor, float]:
    """
    Get the policy and value from the model

    Args:
        model (PolicyValueNetwork): the model to use
        board (Board): the current board state
    """
    model.eval()

    x = encode_board(board)
    x = x.unsqueeze(0)
    x = x.float()

    policy_logits, value = model(x)
    logits = policy_logits[0]

    probs = torch.softmax(logits, dim=0)
    value = value.item()

    return probs, value


def pick_move_minimax(board: Board, to_play: Player, **kwargs: Any) -> tuple[int, int]:
    """Pick the next best move given the board, model, device, and temperature using the Minimax algorithm

    Args:
        board (Board): the current board state
        to_play (Player): the player to play
        **kwargs: additional keyword arguments

    Returns:
        tuple[int, int]: the move
    """
    kw = dict(kwargs)
    stats_out = kw.pop("stats_out", None)
    depth = kw.pop("depth", 3)
    use_alpha_beta = kw.pop("use_alpha_beta", True)
    max_time_ms = kw.pop("max_time_ms", None)
    komi = kw.pop("komi", KOMI)
    rules = kw.pop("rules", RULES)

    deadline = None
    if max_time_ms is not None and float(max_time_ms) > 0:
        deadline = time.perf_counter() + float(max_time_ms) / 1000.0

    best_move = next_best_move(
        board,
        to_play.get_color() == WHITE_COLOR,
        depth=depth,
        use_alpha_beta=use_alpha_beta,
        komi=komi,
        rules=rules,
        deadline=deadline,
        stats_out=stats_out,
    )
    return best_move
