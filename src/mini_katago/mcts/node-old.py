import math
from typing import Self

from mini_katago.go.move import Move
from mini_katago.go.player import Player
from mini_katago.misc.constants import EXPLORATION_CONSTANT, INFINITY


class Node:
    """
    A node represents a game state (board position).
    """

    def __init__(
        self,
        visits: int,
        total_wins: int,
        player_to_play: Player,
        parent: Self | None,
        move_from_parent: Move | None,
    ) -> None:
        """
        Initialize a node object

        Args:
            visits (int): the amount of visits that the node has
            total_wins (int): the accumulated wins of this node
            player_to_play (Player): the player that is about to play next
            parent (Self | None): pointer to the previous node, root has None
            move_from_parent (Move | None): the parent move that leads to this node, root has None
        """
        self.visits = visits
        self.total_wins = total_wins
        self.player_to_play = player_to_play
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.untried_moves: list[Move] | None = None
        self.children: dict[Move, Self] = {}

    def uct_score(self, parent_visits: int, C: float = EXPLORATION_CONSTANT) -> float:
        """
        Calculate the UCT (Upper Confidence Bound applied to Trees) score

        Args:
            parent_visits (int): the node's parent's visits
            C (float, optional): the exploration constant, normally between 1.2-2. Defaults to 1.5.

        Returns:
            float: the UCT score
        """
        if self.visits == 0:
            return INFINITY
        return self.total_wins / self.visits + C * math.sqrt(
            math.log(max(1, parent_visits))  # uses max(1, parent_visits) as a safeguard
            / self.visits
        )

    def select_child(self) -> Self | None:
        """
        Return the child with the highest UCT score

        Returns:
            Self | None: the child with the highest UCT score, or None if node has no children
        """
        best_score = -INFINITY
        best_node: Self | None = None
        if self.children:
            for node in self.children.values():
                score = node.uct_score(node.parent.visits)  # type: ignore
                if score > best_score:
                    best_score = score
                    best_node = node
            return best_node
        return None

    def __repr__(self) -> str:
        """
        Return a developer-friendly message for debugging

        Returns:
            str: a developer friendly message
        """
        return f"visits: {self.visits}, total_wins: {self.total_wins}, player_to_player: {self.player_to_play}, parent: {self.parent}, move_from_parent: {self.move_from_parent}"
