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
) -> tuple[tuple[int, int] | None, float]:
    model.eval()

    x = utils.encode_board(board)
    x = x.unsqueeze(0).to(device)

    policy_logits, _value = model(x)
    logits = policy_logits[0]

    if temperature <= 0.0:
        probs = torch.softmax(logits, dim=0)
    else:
        probs = torch.softmax(logits / temperature, dim=0)

    order = torch.argsort(probs, descending=True).tolist()

    for idx in order:
        if idx == PASS_INDEX:
            move_pos = None
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
                return move_pos, probs[idx].item()

    return None, probs[idx].item()


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

        best = pick_move(model=model, board=board, device=device)
        if best[0] is not None:
            board.place_move(best[0], WHITE_COLOR)
        else:
            board.pass_move()
        board.print_ascii_board()
