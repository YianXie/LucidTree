# mini_katago/mcts/node.py

import math
from typing import Self, Optional

from mini_katago.constants import EXPLORATION_CONSTANT
from mini_katago.go.board import Board
from mini_katago.go.move import Move
from mini_katago.go.player import Player


class Node:
    """
    A node represents a game state (board position).

    Key changes:
    - Don't use Move objects as dict keys (they may be mutable); use (row,col) tuples.
    - Support PASS as a legal child using a sentinel position (-1, -1).
    - Incremental expansion: expand ONE move per visit instead of expanding all at once.
    - Keep API similar to your existing code.
    """

    PASS_POS = (-1, -1)

    def __init__(
        self,
        *,
        prior: float,
        player_to_play: Player,
        parent: Self | None,
        move_from_parent: Move | None,
    ) -> None:
        self.prior = prior
        self.player_to_play = player_to_play
        self.parent = parent
        self.move_from_parent = move_from_parent  # the Move that led here (may be None at root)

        self.is_expanded = False
        self.visits = 0
        self.total_value = 0.0

        # untried moves are stored as positions for safety (immutable keys)
        self.untried_moves: list[tuple[int, int]] | None = None

        # children keyed by position; value is (representative Move or None for PASS, child Node)
        self.children: dict[tuple[int, int], tuple[Optional[Move], "Node"]] = {}

    @property
    def value(self) -> float:
        return self.total_value / max(1, self.visits)

    @property
    def fully_expanded(self) -> bool:
        return self.is_expanded and self.untried_moves is not None and len(self.untried_moves) == 0

    def _init_untried(self, board: Board) -> None:
        if self.untried_moves is not None:
            return

        legal_moves = board.get_legal_moves(self.player_to_play.get_color()) or []
        positions = [mv.get_position() for mv in legal_moves]

        # In Go, PASS is always legal.
        positions.append(self.PASS_POS)

        self.untried_moves = positions
        self.is_expanded = True

    def expand(self, board: Board) -> None:
        """
        Keep this method name so your current call sites don't break.
        But we now do ONE-step expansion setup (not expanding all children immediately).
        """
        self._init_untried(board)

    def expand_one(self, board: Board) -> "Node" | None:
        """
        Expand exactly one child from untried moves.
        Returns the created child node or None if no move to expand.
        """
        self._init_untried(board)
        assert self.untried_moves is not None

        if not self.untried_moves:
            return None

        pos = self.untried_moves.pop()

        # Find representative Move object (optional) for compatibility.
        rep_move: Optional[Move] = None
        if pos != self.PASS_POS:
            # try to find matching Move from legal moves (safe: doesn't rely on Move hashing)
            legal_moves = board.get_legal_moves(self.player_to_play.get_color()) or []
            for mv in legal_moves:
                if mv.get_position() == pos:
                    rep_move = mv
                    break

        # Uniform prior for now (replace later with policy net)
        denom = max(1, (len(self.children) + len(self.untried_moves) + 1))
        child_prior = 1.0 / denom

        child = Node(
            prior=child_prior,
            player_to_play=self.player_to_play.opponent,
            parent=self,
            move_from_parent=rep_move,
        )
        self.children[pos] = (rep_move, child)
        return child

    def puct_score(self, C: float = EXPLORATION_CONSTANT) -> float:
        assert self.parent is not None
        parent_visits = max(1, self.parent.visits)
        return self.value + C * self.prior * (math.sqrt(parent_visits) / (self.visits + 1))

    def select_child(self) -> tuple[tuple[int, int], Self]:
        """
        Return the child with the highest PUCT score.

        Returns:
            (pos, node)
            where pos is (row,col) or PASS_POS (-1,-1)
        """
        if not self.children:
            raise RuntimeError("No children to select from")

        best_pos, (_, best_node) = max(self.children.items(), key=lambda kv: kv[1][1].puct_score())
        return best_pos, best_node
