import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from mini_katago import utils
from mini_katago.constants import SHARD_SIZE
from mini_katago.go.board import Board
from mini_katago.go.game import Game
from mini_katago.nn.datasets.sgf_parser import parse_sgf_file
from mini_katago.nn.split import split_game


logger = utils.setup_logger(name="dataset", log_file="dataset.log", level=logging.INFO)
MAX_GAMES = 10_000


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
            for transformed_board in (game.board, *utils.transform_board(game.board)):
                board = Board(
                    game.board.get_size(),
                    game.black_player,
                    game.white_player,
                )

                moves = transformed_board.get_all_moves()
                for move in moves[: min(self.MAX_MOVES, len(moves))]:
                    to_play = board.get_current_player()

                    x = utils.encode_board(board)
                    move_position = move.get_position()
                    y_policy = utils.move_to_index(move_position)

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
                        board.place_move(move_position, to_play.get_color())

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

    saved_count = 0
    for start in range(0, n_positions, SHARD_SIZE):
        end = min(start + SHARD_SIZE, n_positions)
        x_np = dataset.X[start:end].cpu().numpy()
        y_policy_np = dataset.y_policy[start:end].cpu().numpy()
        y_value_np = dataset.y_value[start:end].cpu().numpy()

        shard_path = output_dir / f"{saved_count:03d}.npz"
        np.savez_compressed(
            shard_path, X=x_np, y_policy=y_policy_np, y_value=y_value_np
        )
        saved_count += 1

        logger.info("Saved shard %d successfully.", saved_count)

    return saved_count


if __name__ == "__main__":
    games: list[Game] = []
    root = utils.get_project_root()
    path = root / "data/raw/sgf/19x19"

    logger.info("Start parsing %d games.", MAX_GAMES)

    games_parsed = 0
    for idx, sgf_file in enumerate(path.glob("*.sgf")):
        try:
            game = parse_sgf_file(sgf_file)
            games.append(game)
            games_parsed += 1
        except ValueError as e:
            logger.warning("Skipped file: %s. ValueError: %s", sgf_file, e)
        except Exception as e:
            logger.warning("Skipped file: %s. Exception: %s", sgf_file, e)

        if idx % 1000 == 0:
            logger.info("%d sgf files parsed", idx)
        if idx == MAX_GAMES:
            logger.info(
                "Attempted to parse %d games. Parsed %d games.", MAX_GAMES, games_parsed
            )
            break

    if games_parsed < MAX_GAMES:
        logger.warning(
            "Some games may not be successfully parsed. Expecting %d | Received %d.",
            MAX_GAMES,
            games_parsed,
        )

    train_games, val_games, test_games = split_game(games)
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
