import copy

from mini_katago.constants import PASS_MOVE_POSITION
from mini_katago.go.board import Board
from mini_katago.go.player import Player
from mini_katago.mcts.node import Node
from mini_katago.nn.agent import load_model
from mini_katago.utils import index_to_row_col


class MCTS:
    def __init__(self) -> None:
        """
        Initialize a Monte Carlo Tree Search program
        """
        self.model = load_model()

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
            while node.is_expanded:
                child_action = node.select_action()
                path.append((node, child_action))

                child = node.children[child_action]
                if child is None:
                    next_board = copy.deepcopy(node.board)
                    next_player = node.to_play.opponent

                    pos = index_to_row_col(child_action)
                    if pos == PASS_MOVE_POSITION:
                        next_board.pass_move()
                    else:
                        next_board.place_move(pos, next_player.get_color())  # type: ignore
                    child = Node(board=next_board, to_play=next_player)  # type: ignore
                    node.children[child_action] = child

                node = child

                if not node.is_expanded:
                    break

            # Expansion + "Simulation" (nn value)
            value = node.expand(self.model)

            # Backup
            self._backup(path, -value)

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
            parent.W[action] += value
            value = -value
