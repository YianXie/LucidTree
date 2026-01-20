# fmt: off

import random
from typing import Any

import torch

from mini_katago.constants import (BLACK_COLOR, BOARD_SIZE, CHANNEL_SIZE,
                                   PASS_INDEX, PASS_MOVE_POSITION, WHITE_COLOR)
from mini_katago.go.board import Board

# fmt: on


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


def encode_board(board: Board) -> torch.Tensor:
    """
    Encode the board to a PyTorch tensor with 6 channels

    Args:
        board (Board): the board to encode

    Returns:
        torch.Tensor: the resulting tensor
    """
    x = torch.zeros(CHANNEL_SIZE, board.size, board.size, dtype=torch.float32)

    for i in range(board.size):
        for j in range(board.size):
            if board.get_move_at_position((i, j)).get_color() == BLACK_COLOR:
                x[0, i, j] = 1  # Black
            elif board.get_move_at_position((i, j)).get_color() == WHITE_COLOR:
                x[1, i, j] = 1  # White
            else:
                x[2, i, j] = 1  # Empty

    # Current player
    if board.get_current_player().get_color() == BLACK_COLOR:
        x[3].fill_(1)

    # Last move
    last_move = board.get_last_move()
    if last_move is not None and not last_move.is_passed():
        x[4, last_move.get_position()] = 1

    # Ko point
    ko_position = board.get_ko_point()
    if ko_position is not None:
        x[5, ko_position] = 1

    return x


def move_to_index(move_position: tuple[int, int]) -> int:
    """
    Calculate the index of a move within the encoded tensor

    Args:
        move_position (tuple[int, int]): the move's position

    Returns:
        int: the index within the tensor
    """
    if move_position == PASS_MOVE_POSITION:
        return PASS_INDEX

    row, col = move_position
    return row * BOARD_SIZE + col
