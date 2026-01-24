from dataclasses import dataclass
from typing import Any

from mini_katago.nn.split import split_game
import torch
from torch.utils.data import Dataset

from mini_katago import utils
from mini_katago.constants import BLACK_COLOR, USE_VALUE
from mini_katago.go.board import Board
from mini_katago.go.game import Game
from mini_katago.misc.sgf_parser import parse_sgf_file


@dataclass
class Sample:
    """
    A sample dataclass consisting of the channels, policy network, and value network
    """

    x: torch.Tensor
    policy_y: int
    value_y: float | None


class SgfPolicyValueDataset(Dataset[Any]):
    def __init__(self, games: list[Game], use_value: bool = USE_VALUE) -> None:
        """
        Initialize a dataset from the given games

        Args:
            games (list[Game]): a list of games
            use_value (bool, optional): whether to calculate value network or not. Defaults to USE_VALUE.
        """
        self.use_value = use_value

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

                    if use_value:
                        if winner is None:
                            ys_value.append(0.0)
                        else:
                            win_color = 1 if winner.get_color() == BLACK_COLOR else -1
                            ys_value.append(
                                1.0 if win_color == to_play.get_color() else -1.0
                            )

                    if move.is_passed():
                        board.pass_move()
                    else:
                        board.place_move(move_position, to_play.get_color())

        self.X: torch.Tensor = torch.stack(xs, dim=0)
        self.y_policy: torch.Tensor = torch.tensor(ys_policy, dtype=torch.long)
        self.y_value: torch.Tensor | None = None

        if use_value:
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


if __name__ == "__main__":
    games: list[Game] = []
    root = utils.get_project_root()
    path = root / "data/raw/sgf"

    for sgf_file in path.iterdir():
        try:
            game = parse_sgf_file(sgf_file)
            games.append(game)
        except ValueError as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"Skipped game. Error: {e}")

    train_games, val_games, test_games = split_game(games)
    train_dataset = SgfPolicyValueDataset(train_games, use_value=USE_VALUE)
    val_dataset = SgfPolicyValueDataset(val_games, use_value=USE_VALUE)
    test_dataset = SgfPolicyValueDataset(test_games, use_value=USE_VALUE)

    torch.save(
        {
            "X": train_dataset.X,
            "y_policy": train_dataset.y_policy,
            "y_value": train_dataset.y_value,
        },
        root / "data/processed/go_9x9_train.pt",
    )
    torch.save(
        {
            "X": val_dataset.X,
            "y_policy": val_dataset.y_policy,
            "y_value": val_dataset.y_value,
        },
        root / "data/processed/go_9x9_val.pt",
    )
    torch.save(
        {
            "X": test_dataset.X,
            "y_policy": test_dataset.y_policy,
            "y_value": test_dataset.y_value,
        },
        root / "data/processed/go_9x9_test.pt",
    )
