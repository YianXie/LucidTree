"""
A combination of Monte Carlo Tree Search and Neural Network
"""

import copy
import random

# fmt: off
from mini_katago.go.board import Board
from mini_katago.go.move import Move
from mini_katago.go.player import Player
from mini_katago.mcts.node import Node
from mini_katago.misc.constants import (ADJ_BOOST, BLACK_COLOR, CAPTURE_BOOST,
                                        MAX_GAME_DEPTH, NUM_SIMULATIONS,
                                        WHITE_COLOR)

# fmt: on


def weighted_choice(moves: list[Move], weights: list[float]) -> Move:
    """
    Select a biased-random move based on the given weights and moves

    Args:
        moves (list[Move]): a list of legal moves
        weights (list[float]): a list of weights corresponding to the list of moves

    Returns:
        Move: the selected move
    """
    return random.choices(moves, weights=weights, k=1)[0]


def move_weight(
    board: Board,
    move: Move,
    color: int,
    *,
    capture_boost: float = CAPTURE_BOOST,
    adj_boost: float = ADJ_BOOST,
) -> float:
    """
    Weight a move based on its capture count and whether it is connected to own stones

    Args:
        board (Board): the board to check the move on
        move (Move): the move to check the weight of
        color (int): the color of the player to check the move for
        capture_boost (float, optional): the boost for captures. Defaults to 6.0.
        adj_boost (float, optional): the boost for adjacent stones. Defaults to 1.8.

    Returns:
        float: the weight of the move
    """
    weight = 1.0
    prev_color = move.get_color()
    move.set_color(color)
    captures = board.check_captures(move)
    if captures:
        weight *= len(captures) ** capture_boost

    neighbors = board.get_neighbors(move)
    for neighbor in neighbors:
        if neighbor.get_color() == move.color:
            weight *= adj_boost
            break

    move.set_color(prev_color)
    return weight


def semi_random_move(board: Board, legal_moves: list[Move], color: int) -> Move:
    """
    Select a move semi-randomly based on the weights of the moves

    Args:
        board (Board): the board to check the move on
        legal_moves (list[Move]): the legal moves to select from
        color (int): the color of the player to select the move for

    Returns:
        Move: the semi-randomly selected move
    """
    weights = [move_weight(board, move, color) for move in legal_moves]
    return weighted_choice(legal_moves, weights)


class MCTS:
    """
    Monte Carlo Tree Search
    """

    def run(
        self,
        root_board: Board,
        root_player: Player,
        num_simulations: int = NUM_SIMULATIONS,
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

            # 3) Rollout (temporary)
            depth = 0
            rollout_player = player
            while not board.is_terminate() and depth < MAX_GAME_DEPTH:
                legal_moves = board.get_legal_moves(rollout_player.get_color())
                depth += 1
                if not legal_moves:
                    board.pass_move()
                    continue
                move = semi_random_move(board, legal_moves, rollout_player.get_color())
                board.place_move(move.get_position(), rollout_player.get_color())

            # 4) Back-propagate
            # TODO: replace with CNN value network later
            black_score, white_score = board.calculate_score()
            value = (
                1
                if (
                    (player.get_color() == BLACK_COLOR and black_score > white_score)
                    or (player.get_color() == WHITE_COLOR and white_score > black_score)
                )
                else -1
            )
            self._backup(search_path, value)

        return root

    def _backup(self, search_path: list[Node], value: float) -> None:
        for node in reversed(search_path):
            node.visits += 1
            node.total_value += value
            value = -value


# Demo
if __name__ == "__main__":
    mcts = MCTS()
    black_player, white_player = (
        Player("Black Player", BLACK_COLOR),
        Player("White Player", WHITE_COLOR),
    )
    black_player.opponent, white_player.opponent = white_player, black_player
    board = Board(9, black_player, white_player)
    while not board.is_terminate():
        row, col = map(int, input("Enter a position to play: ").split())
        board.place_move((row, col), BLACK_COLOR)
        board.print_ascii_board()

        result = mcts.run(board, white_player, 100)
        print(result.visits)
        board.print_ascii_board()
