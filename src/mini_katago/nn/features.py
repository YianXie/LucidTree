import torch

from mini_katago.constants import BLACK_COLOR, CHANNEL_SIZE, WHITE_COLOR
from mini_katago.go.board import Board


def encode_board(board: Board) -> torch.Tensor:
    """
    Encode the board to a PyTorch tensor with 6 channels

    Args:
        board (Board): the board to encode

    Returns:
        torch.Tensor: the resulting tensor
    """
    x = torch.zeros(CHANNEL_SIZE, board.size, board.size, dtype=torch.int16)

    # Direct access to board state instead of repeated function calls
    for i in range(board.size):
        for j in range(board.size):
            color = board.state[i][j].get_color()
            if color == BLACK_COLOR:
                x[0, i, j] = 1  # Black
            elif color == WHITE_COLOR:
                x[1, i, j] = 1  # White
            else:
                x[2, i, j] = 1  # Empty

    # Current player
    if board.get_current_player().get_color() == BLACK_COLOR:
        x[3].fill_(1)

    # Last move
    last_move = board.get_last_move()
    if last_move is not None and not last_move.is_passed():
        row, col = last_move.get_position()
        x[4, row, col] = 1

    # Ko point
    ko_position = board.get_ko_point()
    if ko_position is not None:
        x[5, ko_position[0], ko_position[1]] = 1

    return x
