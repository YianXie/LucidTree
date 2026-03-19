# fmt: off

from typing import Any

import torch
import torch.nn as nn

from lucidtree.common.paths import get_project_root
from lucidtree.constants import PASS_INDEX, PASS_MOVE_POSITION, WHITE_COLOR
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
    model_name: str | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Load the Neural Network model

    Args:
        model_name (str | None, optional): the name of the model to load. Defaults to None.
            If None, loads the default model from the models directory.
        device (torch.device | None, optional): the device to load the model onto.
            If None, loads to CPU. Use CUDA when available for GPU inference.

    Returns:
        nn.Module: the loaded model
    """
    if model_name is None:
        path = root / "models/checkpoint_19x19.pt"
    else:
        path = root / "models" / f"{model_name}.pt"

    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = PolicyValueNetwork()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model


@torch.no_grad()
def pick_move_nn(
    model: nn.Module, board: Board, device: torch.device, temperature: float = 0.0
) -> tuple[tuple[int, int], float, float]:
    """
    Pick the next best move given the board, model, device, and temperature

    Args:
        model (nn.Module): the model to use
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


@torch.no_grad()
def pick_move_mcts(board: Board, to_play: Player, **kwargs: Any) -> tuple[int, int]:
    """
    Pick the next best move given the board, model, device, and temperature using the MCTS algorithm

    Args:
        board (Board): the current board state
        to_play (Player): the player to play
        **kwargs: additional keyword arguments

    Returns:
        tuple[int, int]: the move
    """
    from lucidtree.mcts.search import MCTS

    mcts = MCTS()
    root = mcts.run(board=board, to_play=to_play, **kwargs)

    pos = MCTS.pick_best_move_position(root)
    return pos


def pick_move_minimax(board: Board, to_play: Player, **kwargs: Any) -> tuple[int, int]:
    """Pick the next best move given the board, model, device, and temperature using the Minimax algorithm

    Args:
        board (Board): the current board state
        to_play (Player): the player to play
        **kwargs: additional keyword arguments

    Returns:
        tuple[int, int]: the move
    """
    depth = kwargs.get("depth", 2)
    best_move = next_best_move(board, to_play.get_color() == WHITE_COLOR, depth=depth)
    return best_move
