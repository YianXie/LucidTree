# fmt: off

import logging
import random
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_katago.constants import (BLACK_COLOR, BOARD_SIZE, CHANNEL_SIZE,
                                   PASS_INDEX, PASS_MOVE_POSITION, WHITE_COLOR)
from mini_katago.go.board import Board
from mini_katago.go.player import Player

# fmt: on


def get_project_root() -> Path:
    """
    Find the project root by searching for a .git directory or a pyproject.toml file

    Raises:
        FileNotFoundError: if root could not be found

    Returns:
        Path: the path to start with
    """
    current_file_path = Path(__file__).resolve()
    for parent in current_file_path.parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent

    # Fallback or error handling if the root isn't found
    raise FileNotFoundError(
        "Project root could not be found based on standard markers."
    )


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
    x = torch.zeros(CHANNEL_SIZE, board.size, board.size, dtype=torch.uint8)

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
        x[5, ko_position] = 1

    return x


def move_to_index(move_position: tuple[int, int] | None, /) -> int:
    """
    Calculate the index of a move within the encoded tensor

    Args:
        move_position (tuple[int, int] | None): the move's position

    Returns:
        int: the index within the tensor
    """
    if move_position == PASS_MOVE_POSITION or move_position is None:
        return PASS_INDEX

    row, col = move_position
    return row * BOARD_SIZE + col


def index_to_row_col(index: int, /) -> tuple[int, int]:
    """
    Convert a board index to a move position

    Args:
        index (int): the move index
    """
    if index == PASS_INDEX:
        return (-1, -1)

    return divmod(index, BOARD_SIZE)


def transform_board(board: Board) -> list[Board]:
    """
    Transform the board with rotation and reflection

    Args:
        board (Board): the board to transform

    Returns:
        tuple[Board, ...]: the resulting boards
    """
    rotated_clockwise_board_90 = Board(
        board.get_size(),
        Player(board.get_black_player().get_name(), BLACK_COLOR),
        Player(board.get_white_player().get_name(), WHITE_COLOR),
    )
    rotated_counterclockwise_board_90 = Board(
        board.get_size(),
        Player(board.get_black_player().get_name(), BLACK_COLOR),
        Player(board.get_white_player().get_name(), WHITE_COLOR),
    )
    rotated_board_180 = Board(
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
            rotated_clockwise_board_90.pass_move()
            rotated_counterclockwise_board_90.pass_move()
            rotated_board_180.pass_move()
            reflected_x_board.pass_move()
            reflected_y_board.pass_move()
        else:
            # This is a place move
            row, col = move.get_position()
            color = move.get_color()

            # Transform coordinates for each transformation
            # Rotate 90 degrees clockwise: (row, col) -> (col, n - row - 1)
            new_row, new_col = col, n - row - 1
            rotated_clockwise_board_90.place_move((new_row, new_col), color)

            # Rotate 90 degrees counterclockwise: (row, col) -> (n - col - 1, row)
            new_row, new_col = n - col - 1, row
            rotated_counterclockwise_board_90.place_move((new_row, new_col), color)

            # Rotate 180 degrees: (row, col) -> (n - row - 1, n - col - 1)
            new_row, new_col = n - row - 1, n - col - 1
            rotated_board_180.place_move((new_row, new_col), color)

            # Reflect across x-axis (horizontal): (row, col) -> (n - row - 1, col)
            new_row, new_col = n - row - 1, col
            reflected_x_board.place_move((new_row, new_col), color)

            # Reflect across y-axis (vertical): (row, col) -> (row, n - col - 1)
            new_row, new_col = row, n - col - 1
            reflected_y_board.place_move((new_row, new_col), color)

    boards = [
        rotated_clockwise_board_90,
        rotated_counterclockwise_board_90,
        rotated_board_180,
        reflected_x_board,
        reflected_y_board,
    ]
    random.shuffle(boards)

    return boards[:2]


def setup_logger(
    name: str, log_file: Path | str, level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger instance

    Args:
        name (str): the name of the logger
        log_file (Path | str): the name of the .log file
        level (int, optional): the default logger level. Defaults to logging.INFO.

    Returns:
        logging.Logger: the logger
    """
    root = get_project_root() / "logs"
    path = root / log_file
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(path, maxBytes=10_000_000, backupCount=5)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_npz_dataset(path: Path | str) -> dict[str, Any]:
    """
    Load a npz dataset from a given path

    Args:
        path (Path | str): the path to the .npz file

    Returns:
        dict[str, Any]: the data in the dataset
    """
    data = np.load(path, mmap_mode="r")
    return {
        "X": data["X"],
        "y_policy": data["y_policy"],
        "y_value": data["y_value"],
    }
