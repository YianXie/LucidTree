import copy

import numpy as np
import torch

from mini_katago.constants import BLACK_COLOR, KOMI, PASS_INDEX
from mini_katago.go.board import Board
from mini_katago.go.player import Player
from mini_katago.mcts.node import Node
from mini_katago.nn.agent import load_model
from mini_katago.utils import index_to_row_col


class MCTS:
    """
    A Monte Carlo Tree Search algorithm
    """

    def __init__(self) -> None:
        """
        Initialize a Monte Carlo Tree Search program
        """
        self.model = load_model()
        self.model.eval()

    @torch.no_grad()
    def run(self, board: Board, to_play: Player, num_simulations: int = 1000) -> Node:
        """
        Run the MCTS on the current board

        Args:
            board (Board): the current board
            to_play (Player): the current player to play
            num_simulations (int, optional): the amount of simulations to run. Defaults to 1000.

        Returns:
            Node: the searched root
        """
        if to_play.opponent is None or to_play.opponent.opponent is None:
            raise RuntimeError("Player argument missing `opponent` attribute")

        root = Node(board=board, to_play=to_play)
        root.expand(self.model)

        for _ in range(num_simulations):
            node = root
            path: list[tuple[Node, int]] = []

            # Selection
            while node.is_expanded and not node.board.is_terminate():
                child_action = node.select_action()
                path.append((node, child_action))

                child = node.children[child_action]
                if child is None:
                    move_color = node.to_play.get_color()
                    next_board = copy.deepcopy(node.board)

                    if child_action == PASS_INDEX:
                        next_board.pass_move()
                    else:
                        next_board.place_move(
                            index_to_row_col(child_action), move_color
                        )

                    next_player = node.to_play.opponent
                    child = Node(board=next_board, to_play=next_player)  # type: ignore
                    node.children[child_action] = child

                node = child

                if not node.is_expanded:
                    break

            # Expansion + "Simulation" (nn value)
            if node.is_expanded:
                # Node is already expanded (terminal node case)
                # Recompute the value based on game outcome
                black_score, white_score = node.board.calculate_score()
                black_final = black_score
                white_final = white_score + KOMI
                
                if black_final > white_final:
                    result = 1.0  # Black wins
                elif black_final < white_final:
                    result = -1.0  # Black loses
                else:
                    result = 0.0  # Draw
                
                # Return value from current player's perspective
                value = result if node.to_play.get_color() == BLACK_COLOR else -result
            else:
                value = node.expand(self.model)

            # Backup
            self._backup(path, value)

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
    def pick_move(root: Node) -> tuple[int, int]:
        """
        Pick a move position based on the highest visit count

        Args:
            root (Node): the searched root node

        Returns:
            tuple[int, int]: the position
        """
        legal = root.legal_mask
        idx = int(np.argmax(np.where(legal, root.N, -1)))
        return index_to_row_col(int(idx))
