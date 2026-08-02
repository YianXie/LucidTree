import numpy as np
import torch

from lucidtree.constants import BLACK_COLOR, CHANNEL_SIZE, WHITE_COLOR
from lucidtree.go.board import Board


def encode_board(board: Board) -> torch.Tensor:
    """
    Encode the board to a PyTorch tensor with 6 channels

    Args:
        board (Board): the board to encode

    Returns:
        torch.Tensor: the resulting tensor
    """
    x = np.zeros((CHANNEL_SIZE, board.size, board.size), dtype=np.int16)

    # Read the grid once into an array rather than assigning into the tensor
    # one point at a time; the plane masks are then a single pass each.
    colors = np.array(
        [[move.color for move in row] for row in board.state], dtype=np.int16
    )
    black = colors == BLACK_COLOR
    white = colors == WHITE_COLOR
    x[0] = black  # Black
    x[1] = white  # White
    x[2] = ~(black | white)  # Empty

    # Current player
    if board.get_current_player().get_color() == BLACK_COLOR:
        x[3] = 1

    # Last move
    last_move = board.get_last_move()
    if last_move is not None and not last_move.is_passed():
        row, col = last_move.get_position()
        x[4, row, col] = 1

    # Ko point
    ko_position = board.get_ko_point()
    if ko_position is not None:
        x[5, ko_position[0], ko_position[1]] = 1

    return torch.from_numpy(x)


def value_to_winrate(value: float, color: int) -> dict[str, float]:
    """
    Convert a nn value to human-readable winrate percentage

    Args:
        value (float): the raw value
        color (int): the current player's color
    """
    black_winrate = (value + 1) / 2 * 100
    if color == WHITE_COLOR:
        black_winrate = 100.0 - black_winrate
    white_winrate = 100.0 - black_winrate
    return {"black": black_winrate, "white": white_winrate}
