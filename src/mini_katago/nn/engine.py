# fmt: off

import torch
import torch.nn as nn

from mini_katago import utils
from mini_katago.constants import (BLACK_COLOR, BOARD_SIZE, PASS_INDEX,
                                   WHITE_COLOR)
from mini_katago.go.board import Board
from mini_katago.go.move import Move
from mini_katago.go.player import Player
from mini_katago.nn.model import SmallPVNet

# fmt: on


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
    print(x)
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

    # If no valid move found (should not happen in normal gameplay), return pass
    return None, probs[PASS_INDEX].item(), value


if __name__ == "__main__":
    board = Board(
        BOARD_SIZE,
        Player("Black player", BLACK_COLOR),
        Player("White player", WHITE_COLOR),
    )

    root = utils.get_project_root()
    checkpoint = torch.load(root / "models/checkpoint.pt", map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallPVNet()
    model.load_state_dict(checkpoint["model_state_dict"])

    while not board.is_terminate():
        row, col = map(int, input("Enter your move: ").split())
        if row == -1 and col == -1:
            board.show_board()
            break
        board.place_move((row, col), BLACK_COLOR)
        board.print_ascii_board()

        best, prob, value = pick_move(model=model, board=board, device=device)
        print(f"Move probability: {prob}")
        print(f"Value: {value}")

        if best is not None:
            board.place_move(best, WHITE_COLOR)
        else:
            board.pass_move()
        board.print_ascii_board()
