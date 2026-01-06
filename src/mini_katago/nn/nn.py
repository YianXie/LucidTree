"""
A combination of Monte Carlo Tree Search and Neural Network
"""

import copy
import math
from typing import Self

# fmt: off
from mini_katago.go.board import Board, Move
from mini_katago.go.player import Player
from mini_katago.misc.constants import (BLACK_COLOR, EXPLORATION_CONSTANT,
                                        INFINITY, NUM_SIMULATIONS, WHITE_COLOR)

# fmt: on


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
        return self.total_wins / self.visits

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


class MCTS:
    """
    MCTS-related functions
    """

    @staticmethod
    def run(
        root_board: Board, root_player: Player, num_simulations: int = NUM_SIMULATIONS
    ) -> Node:
        """
        A Monte Carlo Tree Search algorithm to find the best move for the root player

        Args:
            root_board (Board): the board to start the search from
            root_player (Player): the player to start the search from
            num_simulations (int, optional): the number of simulations to run. Defaults to 1000.

        Returns:
            Move | None: the best move for the root player
        """
        root = Node(
            prior=0,
            player_to_play=root_player,
            parent=None,
            move_from_parent=None,
        )
        root.expand(root_board)

        for _ in range(num_simulations):
            node = root
            player = root_player
            board = copy.deepcopy(root_board)

            # 1) Selection
            search_path = [node]
            while node.is_expanded:
                move, node = node.select_child()  # type: ignore
                board.place_move(move.get_position(), player.get_color())
                search_path.append(node)
                player = player.opponent

            # 2) Expansion
            if not board.is_terminate():
                node.expand(board)

            # 3) Back-propagate
            black_score, white_score = board.calculate_score()
            while node is not None:
                if (
                    node.player_to_play.get_color() * -1 == BLACK_COLOR
                    and black_score > white_score
                ) or (
                    node.player_to_play.get_color() * -1 == WHITE_COLOR
                    and white_score > black_score
                ):
                    node.total_wins += 1
                node.visits += 1
                node = node.parent  # type: ignore

        return root
