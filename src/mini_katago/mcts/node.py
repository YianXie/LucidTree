from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from mini_katago.constants import BOARD_SIZE, INFINITY
from mini_katago.go.board import Board
from mini_katago.go.player import Player
from mini_katago.utils import encode_board, move_to_index


class Node:
    """
    A node represents a game state (board position).
    """

    total_actions = BOARD_SIZE * BOARD_SIZE + 1
    eps = 1e-8

    def __init__(
        self,
        board: Board,
        to_play: Player,
    ) -> None:
        """
        Initialize a node

        Args:
            board (Board): the current board state
            to_play (Player): the current player to play in this board state
        """
        self.board = board
        self.to_play = to_play

        self.N = np.zeros([self.total_actions], dtype=np.int32)
        self.W = np.zeros([self.total_actions], dtype=np.float32)

        self.children: list[Node | None] = [None] * self.total_actions
        self.legal_mask = np.zeros(self.total_actions, dtype=np.bool_)

        self.P = np.zeros(self.total_actions, dtype=np.float32)
        self.is_expanded = False

    def expand(self, model: nn.Module) -> float:
        """
        Expand the node by computing legal moves

        Args:
            model (nn.Module): the policy/value policy network model
        """
        if self.is_expanded:
            raise RuntimeError("expanded() called on already expanded node")

        legal_moves = self.board.get_legal_moves(self.to_play.get_color())
        for move in legal_moves:
            idx = move_to_index(move.get_position())
            self.legal_mask[idx] = np.True_

        policy_logits, value = model(encode_board(self.board).unsqueeze(0))
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
        total_value_sum = self.W[action]
        total_visit_count = self.N[action] + self.eps
        return float(total_value_sum / total_visit_count)

    def U(self, action: int, c_puct: float = 1.5) -> float:
        """
        Calculate the PUCT score for a given action

        Args:
            action (int): the action
            c_puct (float, optional): the exploration constant. Defaults to math.sqrt(2).

        Returns:
            float: the PUCT score
        """
        sum_visits = self.N.sum()
        prior = self.P[action]
        action_visits = self.N[action]
        return float(
            c_puct * prior * (math.sqrt(sum_visits + 1.0) / (1.0 + action_visits))
        )

    def select_action(self, c_puct: float = 1.5) -> int:
        """
        Select the action with the highest mean value + PUCT score

        Returns:
            int: the action index
        """
        best_score = -INFINITY
        best_action = 0
        for action in range(self.total_actions):
            if not self.legal_mask[action]:
                continue

            score = self.Q(action) + self.U(action, c_puct)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action
