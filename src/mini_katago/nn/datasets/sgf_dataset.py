import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from mini_katago import utils
from mini_katago.go.board import Board
from mini_katago.go.game import Game
from mini_katago.nn.datasets.sgf_parser import parse_sgf_file
from mini_katago.nn.split import split_game


class SgfPolicyValueDataset(Dataset[Any]):
    def __init__(self, games: list[Game], /) -> None:
        """
        Initialize a dataset from the given games

        Args:
            games (list[Game]): a list of games
        """
        xs: list[torch.Tensor] = []
        ys_policy: list[int] = []
        ys_value: list[float] = []

        for game in games:
            winner = game.winner

            # Iterate over augmented version of the game
            for transformed_board in (game.board, *utils.transform_board(game.board)):
                board = Board(
                    game.board.get_size(),
                    game.black_player,
                    game.white_player,
                )

                for move in transformed_board.get_all_moves():
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

        self.X: torch.Tensor = torch.stack(xs, dim=0)
        self.y_policy: torch.Tensor = torch.tensor(ys_policy, dtype=torch.long)
        self.y_value: torch.Tensor | None = None
        self.y_value = torch.tensor(ys_value, dtype=torch.float32)

    def __len__(self) -> int:
        """
        Get the length of the samples

        Returns:
            int: the length of the samples
        """
        return self.X.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """
        Retrieve the sample at a specific index

        Args:
            index (int): the index of the sample

        Returns:
            tuple[torch.Tensor, ...]: the sample at that specific index
        """
        if self.y_value is None:
            return self.X[index], self.y_policy[index]
        return self.X[index], self.y_policy[index], self.y_value[index]


SHARD_SIZE = 20_000
MAX_SAVE_RETRIES = 3
RETRY_DELAY_SECONDS = 2
logger = logging.getLogger(__name__)


def _save_dataset_as_shards(
    dataset: SgfPolicyValueDataset,
    output_dir: Path,
) -> int:
    """Save dataset as shards of SHARD_SIZE positions each.
    Returns the number of shards successfully saved.
    Continues on per-shard errors; logs failures but does not abort.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_positions = len(dataset)
    print(f"Total positions: {n_positions}")
    shard_idx = 0
    saved_count = 0

    for start in range(0, n_positions, SHARD_SIZE):
        end = min(start + SHARD_SIZE, n_positions)
        shard_data = {
            "X": dataset.X[start:end],
            "y_policy": dataset.y_policy[start:end],
        }
        if dataset.y_value is not None:
            shard_data["y_value"] = dataset.y_value[start:end]

        shard_path = output_dir / f"{shard_idx:03d}.pt"
        tmp_path = output_dir / f"{shard_idx:03d}.pt.tmp"

        for attempt in range(MAX_SAVE_RETRIES):
            try:
                torch.save(shard_data, tmp_path)
                tmp_path.rename(shard_path)
                saved_count += 1
                break
            except (RuntimeError, OSError) as e:
                if attempt < MAX_SAVE_RETRIES - 1:
                    logger.warning(
                        "Shard %s save attempt %d failed: %s. Retrying in %ds...",
                        shard_path.name,
                        attempt + 1,
                        e,
                        RETRY_DELAY_SECONDS,
                    )
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logger.error(
                        "Shard %s failed after %d attempts: %s. Skipping.",
                        shard_path.name,
                        MAX_SAVE_RETRIES,
                        e,
                    )
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass

        shard_idx += 1

    return saved_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    games: list[Game] = []
    root = utils.get_project_root()
    path = root / "data/raw/sgf"

    for sgf_file in path.glob("*.sgf"):
        try:
            game = parse_sgf_file(sgf_file)
            games.append(game)
        except ValueError as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"Skipped game. Error: {e}")

    train_games, val_games, test_games = split_game(games)
    train_dataset = SgfPolicyValueDataset(train_games)
    val_dataset = SgfPolicyValueDataset(val_games)
    test_dataset = SgfPolicyValueDataset(test_games)

    train_saved = _save_dataset_as_shards(train_dataset, root / "data/processed/train")
    val_saved = _save_dataset_as_shards(val_dataset, root / "data/processed/val")
    test_saved = _save_dataset_as_shards(test_dataset, root / "data/processed/test")

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
        logger.warning("Some shards failed to save. Check logs above for details.")
