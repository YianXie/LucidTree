import math

import numpy as np

from mini_katago.constants import BOARD_SIZE, INFINITY
from mini_katago.go.player import Player
from mini_katago.typing import BoolArray, FloatArray, IntArray


class Node:
    """
    A node represents a game state (board position).
    """

    total_actions = BOARD_SIZE * BOARD_SIZE + 1
    eps = 1e-8

    def __init__(
        self,
        prior: FloatArray,
        player_to_play: Player,
    ) -> None:
        """
        Initialize a node

        Args:
            prior (FloatArray): the prior probability for selecting each move
            player_to_play (Player): the current player to play in this board state
        """
        self.prior = prior
        self.children_visit_count: IntArray = np.ndarray(
            [self.total_actions], dtype=np.int32
        )
        self.total_value_sum: FloatArray = np.ndarray(
            [self.total_actions], dtype=np.float32
        )
        self.player_to_play = player_to_play

        self.is_expanded = False
        self.children: list[Node | None] = [None] * self.total_actions
        self.legal: BoolArray

    def get_mean_value(self, action: int) -> float:
        """
        Return the mean value of a child index

        Args:
            action (int): the index of the child

        Returns:
            float: the mean value
        """
        total_value_sum = self.total_value_sum[action]
        total_visit_count = self.children_visit_count[action] + self.eps
        return float(total_value_sum / total_visit_count)

    def get_puct_score(self, action: int, C: float = math.sqrt(2)) -> float:
        """
        Calculate the PUCT score for a given action

        Args:
            action (int): the action
            C (float, optional): the exploration constant. Defaults to math.sqrt(2).

        Returns:
            float: the PUCT score
        """
        if not self.legal[action]:
            return -INFINITY

        sum_visits = self.children_visit_count.sum()
        prior = self.prior[action]
        action_visits = self.children_visit_count[action]
        return float(C * prior * (math.sqrt(sum_visits) / (1.0 + action_visits)))

    def select_child(self) -> Node | None:  # noqa: F821
        """
        Select the child with the highest mean value + PUCT score

        Returns:
            Node | None: the child node, or None if there is no children
        """
        if not self.children:
            return None

        best_score = 0.0
        best_node = None
        for i in range(self.total_actions):
            if self.children[i] is None:
                continue
            score = self.get_mean_value(i) + self.get_puct_score(i)
            if score > best_score:
                best_score = score
                best_node = self.children[i]

        return best_node
