import datetime
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from lucidtree.common.logging import setup_logger
from lucidtree.common.paths import get_project_root
from lucidtree.constants import BLACK_COLOR, SHARD_SIZE, WHITE_COLOR
from lucidtree.go.board import Board
from lucidtree.go.coordinates import row_col_to_index
from lucidtree.go.game import Game
from lucidtree.go.player import Player
from lucidtree.nn.datasets.sgf_parser import parse_sgf_files
from lucidtree.nn.features import encode_board
from lucidtree.nn.split import split_game

logger = setup_logger(name="dataset", log_file="dataset.log", level=logging.INFO)
AMOUNT_TO_PARSE = None
START_GAME = 0


def transform_board(board: Board, amount: int = 2) -> list[Board]:
    """
    Transform the board with rotation and reflection

    Args:
        board (Board): the board to transform
        amount (int, optional): the amount of boards to return. Defaults to 2.

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

    return boards[: min(amount, len(boards))]


class SgfPolicyValueDataset(Dataset[Any]):
    """
    A dataset class representing a sgf policy-value network
    """

    MAX_MOVES = 300

    def __init__(self, games: list[Game], /) -> None:
        """
        Initialize a dataset from the given games

        Args:
            games (list[Game]): a list of games
        """
        xs: list[torch.Tensor] = []
        ys_policy: list[int] = []
        ys_value: list[float] = []

        logger.info("SgfPolicyValueDataset received %d games", len(games))
        for idx, game in enumerate(games):
            winner = game.winner

            # Iterate over augmented version of the game
            for transformed_board in (game.board, *transform_board(game.board)):
                board = Board(
                    game.board.get_size(),
                    game.black_player,
                    game.white_player,
                )

                moves = transformed_board.get_all_moves()
                for move in moves[: min(self.MAX_MOVES, len(moves))]:
                    to_play = board.get_current_player()

                    x = encode_board(board)
                    row, col = move.get_position()
                    y_policy = row_col_to_index(row, col)

                    xs.append(x)
                    ys_policy.append(y_policy)

                    if winner is None:
                        ys_value.append(0.0)
                    else:
                        ys_value.append(
                            1.0 if winner.get_color() == to_play.get_color() else -1.0
                        )

                    if move.is_passed():
                        board.pass_move()
                    else:
                        board.place_move((row, col), to_play.get_color())

            if idx % 500 == 0:
                logger.info("%d games parsed", idx)

        self.X: torch.Tensor = torch.stack(xs, dim=0)
        self.y_policy: torch.Tensor = torch.tensor(ys_policy, dtype=torch.int16)
        self.y_value: torch.Tensor = torch.tensor(ys_value, dtype=torch.int8)

    def __len__(self) -> int:
        """
        Get the length of the samples

        Returns:
            int: the length of the samples
        """
        return self.X.size(0)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve the sample at a specific index

        Args:
            index (int): the index of the sample

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the sample at that specific index
        """
        return (self.X[index], self.y_policy[index], self.y_value[index])


def _save_dataset_as_shards(
    dataset: SgfPolicyValueDataset,
    output_dir: Path,
) -> int:
    """
    Save the dataset into multiple different shards

    Args:
        dataset (SgfPolicyValueDataset): the dataset
        output_dir (Path): the directory to save the data

    Returns:
        int: the total shards saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_positions = len(dataset)
    logger.info("n_positions: %d", n_positions)

    saved_shard_count = 0
    shard_idx = 0
    for start in range(0, n_positions, SHARD_SIZE):
        end = min(start + SHARD_SIZE, n_positions)
        x_np = dataset.X[start:end].cpu().numpy()
        y_policy_np = dataset.y_policy[start:end].cpu().numpy()
        y_value_np = dataset.y_value[start:end].cpu().numpy()

        shard_path = output_dir / f"{shard_idx:03d}.npz"
        while shard_path.exists():
            shard_idx += 1
            shard_path = output_dir / f"{shard_idx:03d}.npz"

        np.savez_compressed(
            shard_path, X=x_np, y_policy=y_policy_np, y_value=y_value_np
        )
        saved_shard_count += 1
        shard_idx += 1
        logger.info("Saved shard %d successfully.", saved_shard_count)

    return saved_shard_count


if __name__ == "__main__":
    root = get_project_root()
    path = root / "data/raw/sgf/19x19"

    if AMOUNT_TO_PARSE is None:
        logger.info("Start parsing all games.")
    else:
        logger.info("Start parsing %d games.", AMOUNT_TO_PARSE)
    logger.info("Starting from game %d", START_GAME)
    start_time = time.perf_counter()

    games = parse_sgf_files(
        path,
        start=START_GAME,
        amount=AMOUNT_TO_PARSE,
        log=True,
        logger=logger,
        gap=1000,
    )
    train_games, val_games, test_games = split_game(games, seed=0)
    train_dataset = SgfPolicyValueDataset(train_games)
    val_dataset = SgfPolicyValueDataset(val_games)
    test_dataset = SgfPolicyValueDataset(test_games)

    train_saved = _save_dataset_as_shards(
        train_dataset, root / "data/processed/train/19x19"
    )
    val_saved = _save_dataset_as_shards(val_dataset, root / "data/processed/val/19x19")
    test_saved = _save_dataset_as_shards(
        test_dataset, root / "data/processed/test/19x19"
    )

    total_train = (len(train_dataset) + SHARD_SIZE - 1) // SHARD_SIZE
    total_val = (len(val_dataset) + SHARD_SIZE - 1) // SHARD_SIZE
    total_test = (len(test_dataset) + SHARD_SIZE - 1) // SHARD_SIZE

    logger.info(
        "Saved shards: train %d/%d, val %d/%d, test %d/%d",
        train_saved,
        total_train,
        val_saved,
        total_val,
        test_saved,
        total_test,
    )
    if train_saved < total_train or val_saved < total_val or test_saved < total_test:
        logger.warning("Some shards failed to save.")

    end_time = time.perf_counter()
    duration = end_time - start_time
    logger.info(
        "Total training time: %d seconds, or %s",
        duration,
        datetime.timedelta(seconds=duration),
    )
