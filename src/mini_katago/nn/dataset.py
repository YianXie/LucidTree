from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from mini_katago import utils
from mini_katago.constants import BLACK_COLOR
from mini_katago.go.board import Board
from mini_katago.go.game import Game


@dataclass
class Sample:
    """
    A sample dataclass consisting the channels, policy network, and value network
    """

    x: torch.Tensor
    policy_y: int
    value_y: float | None


class SgfPolicyValueDataset(Dataset[Any]):
    def __init__(
        self, games: list[Game], board_cls: object = Board, use_value: bool = True
    ) -> None:
        """
        Initialize a dataset from the given games

        Args:
            games (list[Game]): a list of games
            board_cls (object, optional): the board class. Defaults to Board.
            use_value (bool, optional): whether to calculate value network or not. Defaults to True.
        """
        self.samples: list[Sample] = []
        for game in games:
            board = Board(game.board.get_size(), game.black_player, game.white_player)
            winner = game.winner

            for move in game.board.get_all_moves():
                to_play = board.get_current_player()

                x = utils.encode_board(board)
                y_policy = utils.move_to_index(move.get_position())

                y_value = None
                if use_value and winner is not None:
                    win_color = 1 if winner.get_color() == BLACK_COLOR else -1
                    y_value = 1.0 if win_color == to_play.get_color() else -1.0

                self.samples.append(Sample(x, y_policy, y_value))

                move_position = move.get_position()
                if move_position is not None:
                    board.place_move(move_position, to_play.get_color())
                else:
                    board.pass_move()

    def __len__(self) -> int:
        """
        Get the length of the samples

        Returns:
            int: the length of the samples
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """
        Retrieve the sample at a specific index

        Args:
            index (int): the index of the sample

        Returns:
            tuple[torch.Tensor, ...]: the sample at that specific index
        """
        sample = self.samples[index]
        if sample.value_y is None:
            return sample.x, torch.tensor(sample.policy_y, dtype=torch.long)
        return (
            sample.x,
            torch.tensor(sample.policy_y, dtype=torch.long),
            torch.tensor(sample.value_y, dtype=torch.float32),
        )
