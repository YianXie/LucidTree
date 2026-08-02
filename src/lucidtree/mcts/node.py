from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from lucidtree.constants import BLACK_COLOR, BOARD_SIZE, INFINITY, KOMI, RULES
from lucidtree.go.board import Board
from lucidtree.go.coordinates import row_col_to_index
from lucidtree.go.player import Player
from lucidtree.nn.features import encode_board


class Node:
    """
    A node represents a game state (board position).
    """

    total_actions = BOARD_SIZE * BOARD_SIZE + 1

    def __init__(
        self,
        board: Board,
        to_play: Player,
        *,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
    ) -> None:
        """
        Initialize a node

        Args:
            board (Board): the current board state
            to_play (Player): the current player to play in this board state
        """
        self.board = board
        self.to_play = to_play
        self.policy_weight = float(policy_weight)
        self.value_weight = float(value_weight)

        self.N = np.zeros([self.total_actions], dtype=np.int32)
        self.W = np.zeros([self.total_actions], dtype=np.float32)

        self.children: list[Node | None] = [None] * self.total_actions
        self.legal_mask = np.zeros(self.total_actions, dtype=np.bool_)

        self.P = np.zeros(self.total_actions, dtype=np.float32)
        self.is_expanded = False

    def expand(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        *,
        komi: float = KOMI,
        rules: str = RULES,
        dirichlet_alpha: float = 0.0,
        dirichlet_epsilon: float = 0.0,
    ) -> float:
        """
        Expand the node by computing legal moves

        Args:
            model (nn.Module): the policy/value policy network model
            device (torch.device | None, optional): the device for inference. Uses model's device if None.
        """
        if self.is_expanded:
            raise RuntimeError("expanded() called on already expanded node")

        # Check if game is over
        if self.board.is_terminate():
            self.is_expanded = True
            # Calculate the game outcome from the current player's perspective
            black_score, white_score = self.board.calculate_score(
                komi=komi, rules=rules
            )

            # Calculate result from Black's perspective
            if black_score > white_score:
                result = 1.0  # Black wins
            elif black_score < white_score:
                result = -1.0  # Black loses
            else:
                result = 0.0  # Draw

            # Return value from current player's perspective
            return result if self.to_play.get_color() == BLACK_COLOR else -result

        legal_moves = self.board.get_legal_moves(self.to_play.get_color())
        for move in legal_moves:
            row, col = move.get_position()
            idx = row_col_to_index(row, col)
            self.legal_mask[idx] = True

        x = encode_board(self.board).unsqueeze(0).float()
        if device is not None:
            x = x.to(device)
        policy_logits, value = model(x)
        probs = (
            torch.softmax(policy_logits[0], dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        probs *= self.legal_mask.astype(np.float32)
        s = float(probs.sum())
        if s > 0:
            probs /= s
        else:
            legal_count = int(self.legal_mask.sum())
            probs[self.legal_mask] = 1.0 / max(1, legal_count)

        if dirichlet_alpha > 0.0 and dirichlet_epsilon > 0.0:
            legal_actions = np.where(self.legal_mask)[0]
            if legal_actions.size > 0:
                noise = np.random.dirichlet(
                    np.full(legal_actions.shape[0], dirichlet_alpha, dtype=np.float32)
                ).astype(np.float32)
                probs[legal_actions] = (1.0 - dirichlet_epsilon) * probs[
                    legal_actions
                ] + dirichlet_epsilon * noise

        self.P = probs
        self.is_expanded = True
        return float(value.item())

    def Q(self, action: int) -> float:
        """
        Return the mean value of a child index

        Args:
            action (int): the index of the child

        Returns:
            float: the mean value
        """
        total_visit_count = self.N[action]
        if total_visit_count == 0:
            return 0.0
        total_value_sum = self.W[action]
        return float(self.value_weight * (total_value_sum / total_visit_count))

    def Q_all(self) -> npt.NDArray[np.float64]:
        """
        Return the mean value of every action at once

        The vectorized form of :meth:`Q`. It reproduces that method's numerics
        exactly rather than approximately: unvisited actions score 0.0, and
        visited actions divide in float64, because `W` is float32 and `N` is
        int32 and NumPy promotes that pair to float64 before dividing. Doing
        the division in float32 instead would round differently and could flip
        a near-tie in :meth:`select_action`.

        Returns:
            npt.NDArray[np.float64]: the mean value of each action
        """
        visited = self.N != 0
        values = np.zeros(self.total_actions, dtype=np.float64)
        np.divide(
            self.W.astype(np.float64),
            self.N.astype(np.float64),
            out=values,
            where=visited,
        )
        np.multiply(values, self.value_weight, out=values, where=visited)
        return values

    def U(self, action: int, c_puct: float = 1.5) -> float:
        """
        Calculate the PUCT score for a given action

        Args:
            action (int): the action
            c_puct (float, optional): the exploration constant. Defaults to 1.5.

        Returns:
            float: the PUCT score
        """
        sum_visits = self.N.sum()
        prior = self.P[action]
        action_visits = self.N[action]
        return float(
            self.policy_weight
            * c_puct
            * prior
            * (math.sqrt(sum_visits) / (1.0 + action_visits))
        )

    def U_all(self, c_puct: float = 1.5) -> npt.NDArray[np.float64]:
        """
        Calculate the PUCT score for every action at once

        The vectorized form of :meth:`U`, and again bit-identical rather than
        merely close. `policy_weight * c_puct` is a Python float, and NumPy
        weak promotion rounds that product to float32 when it meets the
        float32 prior; `np.float32(...)` reproduces that rounding. Widening
        the float32 product to float64 afterwards is exact, and the remaining
        `sqrt(sum) / (1 + N)` factor is float64 in both paths.

        Args:
            c_puct (float, optional): the exploration constant. Defaults to 1.5.

        Returns:
            npt.NDArray[np.float64]: the PUCT score of each action
        """
        sum_visits = self.N.sum()
        scale = np.float32(self.policy_weight * c_puct)
        scores = (scale * self.P).astype(np.float64)
        scores *= math.sqrt(sum_visits) / (1.0 + self.N.astype(np.float64))
        return scores

    def select_action(self, c_puct: float = 1.5) -> np.int64:
        """
        Select the action with the highest mean value + PUCT score

        Illegal actions are pushed to negative infinity and `np.argmax`
        returns the first maximum, so ties resolve to the lowest action index
        exactly as the strict `>` comparison in the original scalar loop did.

        Returns:
            int: the action index
        """
        scores = self.Q_all()
        scores += self.U_all(c_puct)
        np.copyto(scores, -INFINITY, where=~self.legal_mask)

        best_action = np.argmax(scores)
        if not self.legal_mask[best_action]:
            # Nothing is legal at all; the scalar loop never entered its body
            # and fell through to this same index.
            return np.int64(BOARD_SIZE * BOARD_SIZE)
        return np.int64(best_action)
