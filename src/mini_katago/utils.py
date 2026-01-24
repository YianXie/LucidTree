# fmt: off

import random
from typing import Any

import torch

from mini_katago.constants import (BLACK_COLOR, BOARD_SIZE, CHANNEL_SIZE,
                                   PASS_INDEX, PASS_MOVE_POSITION, WHITE_COLOR)
from mini_katago.go.board import Board
from mini_katago.go.player import Player

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


def transform_board(board: Board) -> tuple[Board, Board, Board, Board]:
    """
    Transform the board with rotation and reflection

    Args:
        board (Board): the board to transform

    Returns:
        tuple[Board, Board, Board, Board]: the resulting boards
    """
    rotated_clockwise_board = Board(
        board.get_size(),
        Player(board.get_black_player().get_name(), BLACK_COLOR),
        Player(board.get_white_player().get_name(), WHITE_COLOR),
    )
    rotated_counterclockwise_board = Board(
        board.get_size(),
        Player(board.get_black_player().get_name(), BLACK_COLOR),
        Player(board.get_white_player().get_name(), WHITE_COLOR),
    )
    reflected_x_board = Board(
        board.get_size(),
        Player(board.get_black_player().get_name(), BLACK_COLOR),
        Player(board.get_white_player().get_name(), WHITE_COLOR),
    )
    reflected_y_board = Board(
        board.get_size(),
        Player(board.get_black_player().get_name(), BLACK_COLOR),
        Player(board.get_white_player().get_name(), WHITE_COLOR),
    )

    n = board.get_size()
    for move in board.get_all_moves():
        if move.is_passed():
            rotated_clockwise_board.pass_move()
            rotated_counterclockwise_board.pass_move()
            reflected_x_board.pass_move()
            reflected_y_board.pass_move()
        else:
            # This is a place move
            row, col = move.get_position()
            color = move.get_color()

            # Transform coordinates for each transformation
            # Rotate 90 degrees clockwise: (row, col) -> (col, n - row - 1)
            new_row, new_col = col, n - row - 1
            rotated_clockwise_board.place_move((new_row, new_col), color)

            # Rotate 90 degrees counterclockwise: (row, col) -> (n - col - 1, row)
            new_row, new_col = n - col - 1, row
            rotated_counterclockwise_board.place_move((new_row, new_col), color)

            # Reflect across x-axis (horizontal): (row, col) -> (n - row - 1, col)
            new_row, new_col = n - row - 1, col
            reflected_x_board.place_move((new_row, new_col), color)

            # Reflect across y-axis (vertical): (row, col) -> (row, n - col - 1)
            new_row, new_col = row, n - col - 1
            reflected_y_board.place_move((new_row, new_col), color)

    return (
        rotated_clockwise_board,
        rotated_counterclockwise_board,
        reflected_x_board,
        reflected_y_board,
    )


if __name__ == "__main__":
    # Test transform_board
    base_board = Board(
        BOARD_SIZE,
        Player("Test Black Player", BLACK_COLOR),
        Player("Test White Player", WHITE_COLOR),
    )
    base_board.place_move((1, 2), BLACK_COLOR)
    base_board.place_move((3, 5), WHITE_COLOR)
    print(base_board.get_all_moves())

    for board in transform_board(base_board):
        for move in board.get_all_moves():
            print(move_to_index(move.get_position()))
