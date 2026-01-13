import random
from typing import Any

import torch

from mini_katago.constants import BLACK_COLOR, CHANNEL_SIZE, WHITE_COLOR
from mini_katago.go.board import Board


def weighted_choice(moves: list[Any], weights: list[float]) -> Any:
    """
    Select a biased-random move based on the given weights and moves

    Args:
        moves (list[Move]): a list of legal moves
        weights (list[float]): a list of weights corresponding to the list of moves

    Returns:
        Move: the selected move
    """
    return random.choices(moves, weights=weights, k=1)[0]


def encode_position(board: Board) -> torch.Tensor:
    x = torch.zeros(CHANNEL_SIZE, board.size, board.size, dtype=torch.float32)

    for i in range(board.size):
        for j in range(board.size):
            if board.get_move((i, j)).get_color() == BLACK_COLOR:
                x[0, i, j] = 1  # Black
            elif board.get_move((i, j)).get_color() == WHITE_COLOR:
                x[1, i, j] = 1  # White
            else:
                x[2, i, j] = 1  # Empty

    # Current player
    if board.get_current_player().get_color() == BLACK_COLOR:
        x[3].fill_(1)

    # Last move
    x[4, board.get_nth_move(-1).get_position()] = 1

    # Ko point
    ko_position = board.get_ko_point()
    if ko_position is not None:
        x[5, ko_position] = 1

    return x
