import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lucidtree.constants import BLACK_COLOR, KOMI, PASS_INDEX, RULES
from lucidtree.go.board import Board
from lucidtree.go.coordinates import index_to_row_col
from lucidtree.go.player import Player
from lucidtree.mcts.node import Node
from lucidtree.nn.agent import load_model


class MCTS:
    """
    A Monte Carlo Tree Search algorithm
    """

    def __init__(self, model: Path | str | None = None, **kwargs: Any) -> None:
        """
        Initialize a Monte Carlo Tree Search program

        Args:
            model (Path | str | None, optional): the path to the model to load. Defaults to None.
                If None, loads the default model from the models directory.
                If a string, it is assumed to be the name of the model and is loaded from the models directory.
                If a Path, it is assumed to be the path to the model and is loaded from the given path.
            **kwargs: additional keyword arguments
        """
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = load_model(model=model, device=self.device)
        self.model.eval()
        self.simulations_run = 0

    @torch.no_grad()
    def run(self, board: Board, to_play: Player, **kwargs: Any) -> Node:
        """
        Run the MCTS on the current board

        Args:
            board (Board): the current board
            to_play (Player): the current player to play
            **kwargs: additional keyword arguments
        """
        num_simulations = kwargs.get("num_simulations", 1000)
        c_puct = kwargs.get("c_puct", 1.5)
        komi = kwargs.get("komi", KOMI)
        rules = kwargs.get("rules", RULES)
        dirichlet_alpha = kwargs.get("dirichlet_alpha", 0.0)
        dirichlet_epsilon = kwargs.get("dirichlet_epsilon", 0.0)
        value_weight = kwargs.get("value_weight", 1.0)
        policy_weight = kwargs.get("policy_weight", 1.0)

        if to_play.opponent is None or to_play.opponent.opponent is None:
            raise RuntimeError("Player argument missing `opponent` attribute")

        root = Node(
            board=board,
            to_play=to_play,
            policy_weight=policy_weight,
            value_weight=value_weight,
        )
        root.expand(
            self.model,
            self.device,
            komi=komi,
            rules=rules,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

        max_time_ms = kwargs.get("max_time_ms")
        deadline: float | None = None
        if max_time_ms is not None and float(max_time_ms) > 0:
            deadline = time.perf_counter() + float(max_time_ms) / 1000.0

        simulations_run = 0
        for _ in range(num_simulations):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            simulations_run += 1
            node = root
            path: list[tuple[Node, int]] = []

            # Selection
            while node.is_expanded and not node.board.is_terminate():
                child_action = node.select_action(c_puct=c_puct)
                path.append((node, child_action.item()))

                child = node.children[child_action]
                if child is None:
                    move_color = node.to_play.get_color()
                    next_board = copy.deepcopy(node.board)

                    if child_action.item() == PASS_INDEX:
                        next_board.pass_move()
                    else:
                        next_board.place_move(
                            index_to_row_col(child_action.item()), move_color
                        )

                    next_player = node.to_play.opponent
                    child = Node(
                        board=next_board,
                        to_play=next_player,  # type: ignore
                        policy_weight=policy_weight,
                        value_weight=value_weight,
                    )
                    node.children[child_action] = child

                node = child

                if not node.is_expanded:
                    break

            # Expansion + "Simulation" (nn value)
            if node.is_expanded:
                # Node is already expanded (terminal node case)
                # Recompute the value based on game outcome
                black_score, white_score = node.board.calculate_score(
                    komi=komi, rules=rules
                )

                if black_score > white_score:
                    result = 1.0  # Black wins
                elif black_score < white_score:
                    result = -1.0  # Black loses
                else:
                    result = 0.0  # Draw

                # Return value from current player's perspective
                value = result if node.to_play.get_color() == BLACK_COLOR else -result
            else:
                value = node.expand(self.model)

            # Backup
            self._backup(path, value)

        self.simulations_run = simulations_run
        return root

    def _backup(self, path: list[tuple[Node, int]], value: float) -> None:
        """
        Backup the search path

        Args:
            path (list[tuple[Node, int]]): the search path
            value (float): the value
        """
        for parent, action in reversed(path):
            parent.N[action] += 1
            value = -value
            parent.W[action] += value

    @staticmethod
    def pick_best_move_position(
        root: Node, select_by: str = "visit_count", include_top_moves: int = 1
    ) -> list[tuple[int, int]]:
        """
        Pick a move position based on the highest visit count

        Args:
            root (Node): the searched root node
            select_by (str): the selection method to use
            include_top_moves (int): the number of top moves to include

        Returns:
            list[tuple[int, int]]: the top moves
        """
        legal = root.legal_mask
        if select_by == "value":
            scores = np.array(
                [root.Q(i) if legal[i] else -np.inf for i in range(len(root.N))]
            )
            sorted_scores = np.argsort(scores)[::-1]
            top_moves = sorted_scores[:include_top_moves]
        else:
            sorted_visits = np.argsort(np.where(legal, root.N, -1))[::-1]
            top_moves = sorted_visits[:include_top_moves]

        return [index_to_row_col(int(move)) for move in top_moves]
