import math
from typing import Self

from mini_katago.go.board import Board
from mini_katago.go.move import Move
from mini_katago.go.player import Player
from mini_katago.misc.constants import EXPLORATION_CONSTANT, INFINITY


class Node:
    """
    A node represents a game state (board position).
    """

    def __init__(
        self,
        *,
        prior: float,
        player_to_play: Player,
        parent: Self | None,
        move_from_parent: Move | None,
    ) -> None:
        """
        Initialize a node object

        Args:
            prior (float): the probability of choosing this node from its parent
            player_to_play (Player): the player that is about to play next
            parent (Self | None): pointer to the previous node, root has None
            move_from_parent (Move | None): the parent move that leads to this node, root has None
        """
        self.prior = prior
        self.player_to_play = player_to_play
        self.parent = parent
        self.move_from_parent = move_from_parent

        self.is_expanded = False
        self.visits = 0
        self.total_wins = 0
        self.untried_moves: list[Move] | None = None
        self.children: dict[Move, Node] = {}

    @property
    def value(self) -> float:
        """
        The value of the node (win rate)

        Returns:
            float: the value of the node
        """
        return self.total_wins / max(1, self.visits)  # prevent zero-division error

    def expand(self, board: Board) -> None:
        """
        Expand the node's children to all legal moves available

        Args:
            board (Board): the game board
        """
        legal_moves = board.get_legal_moves(self.player_to_play.get_color())
        for move in legal_moves:
            self.children[move] = Node(
                prior=1 / len(legal_moves),
                player_to_play=self.player_to_play.opponent,
                parent=self,
                move_from_parent=board.get_nth_move(-1),
            )
        self.is_expanded = True

    def puct_score(self, C: float = EXPLORATION_CONSTANT) -> float:
        """
        Calculate the PUCT score

        Args:
            C (float, optional): the exploration constant, normally between 1.2-2. Defaults to 1.5.

        Returns:
            float: the PUCT score
        """
        potential_actions_visits = 0
        for child in self.parent.children.values():  # type: ignore
            potential_actions_visits += child.visits
        return self.value + C * self.prior * (
            math.sqrt(potential_actions_visits) / (self.visits + 1)
        )

    def select_child(self) -> tuple[Move, Self] | None:
        """
        Return the child with the highest PUCT score

        Returns:
            tuple[Move, Self] | None: the child with the highest PUCT score, or None if node has no children
        """
        best_score = -INFINITY
        best_node: Self | None = None
        if self.children:
            for node in self.children.values():
                score = node.puct_score(node.parent.visits)  # type: ignore
                if score > best_score:
                    best_score = score
                    best_node = node  # type: ignore
        return best_node.move_from_parent, best_node  # type: ignore
