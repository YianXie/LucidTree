# fmt: off

from pathlib import Path

import torch
import torch.nn as nn

from mini_katago import utils
from mini_katago.constants import PASS_INDEX
from mini_katago.go.board import Board
from mini_katago.go.move import Move
from mini_katago.nn.model import SmallPVNet

# fmt: on

root = utils.get_project_root()


@torch.no_grad()
def load_model(
    path: Path = root / "models/checkpoint.pt",
    map_location: str = "cpu",
) -> nn.Module:
    """
    Load the Neural Network model

    Args:
        path (Path, optional): the path to the checkpoint file. Defaults to root/"models/checkpoint.pt".
        map_location (str, optional): the map_location for checkpoint. Defaults to "cpu".

    Returns:
        nn.Module: the loaded model
    """
    checkpoint = torch.load(path, map_location=map_location)
    model = SmallPVNet()
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


@torch.no_grad()
def pick_move(
    model: nn.Module, board: Board, device: torch.device, temperature: float = 0.0
) -> tuple[tuple[int, int] | None, float, float]:
    """
    Pick the next best move given the board, model, device, and temperature

    Args:
        model (nn.Module): the model to use
        board (Board): the current board state
        device (torch.device): the PyTorch device
        temperature (float, optional): the temperature. Defaults to 0.0.

    Returns:
        tuple[tuple[int, int] | None, float, float]: the move, probability, and value
    """
    model.eval()

    x = utils.encode_board(board)
    x = x.unsqueeze(0).to(device)

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
            # PASS move is always valid
            return None, probs[idx].item(), value
        else:
            move_pos = utils.index_to_row_col(idx)
            if (
                board.move_is_valid(
                    Move(
                        move_pos[0], move_pos[1], board.get_current_player().get_color()
                    )
                )
                and board.get_move_at_position(move_pos).is_empty()
            ):
                return move_pos, probs[idx].item(), value

    return None, probs[idx].item(), value
