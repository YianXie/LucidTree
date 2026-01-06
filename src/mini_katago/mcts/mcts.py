"""
A pure Monte Carlo Tree Search algorithm for Go
"""

import random

# fmt: off
from mini_katago.go.board import Board
from mini_katago.go.move import Move
from mini_katago.go.player import Player
from mini_katago.misc.constants import (ADJ_BOOST, BLACK_COLOR, CAPTURE_BOOST,
                                        MAX_GAME_DEPTH, NUM_SIMULATIONS,
                                        WHITE_COLOR)

from .node import Node

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


def mcts(
    root_board: Board, root_player: Player, num_simulations: int = NUM_SIMULATIONS
) -> Move | None:
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
        visits=0,
        total_wins=0,
        player_to_play=root_player,
        parent=None,
        move_from_parent=None,
    )
    root.untried_moves = root_board.get_legal_moves(root_player.get_color())

    for _ in range(num_simulations):
        moves_made = 0
        node = root
        player = root_player

        # 1) Selection
        while (
            not node.untried_moves
            and not root_board.is_terminate()
            and len(node.children) > 0
        ):
            temp_node = node.select_child()
            if temp_node is None:
                break
            node = temp_node
            root_board.place_move(
                node.move_from_parent.get_position(),  # type: ignore
                player.get_color(),
            )
            player = player.opponent
            moves_made += 1

        # 2) Expansion (add 1 child)
        if not root_board.is_terminate():
            # Attempt to get legal moves if it not already exists
            if node.untried_moves is None:
                node.untried_moves = root_board.get_legal_moves(player.get_color())

            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                moves_made += 1

                root_board.place_move(move.get_position(), player.get_color())
                child = Node(
                    visits=0,
                    total_wins=0,
                    player_to_play=player.opponent,
                    parent=node,
                    move_from_parent=move,
                )
                player = player.opponent
                child.untried_moves = root_board.get_legal_moves(player.get_color())
                node.children[move] = child
                node = child

        # 3) Simulation (rollout)
        rollout_player = player
        depth = 0
        while not root_board.is_terminate() and depth < MAX_GAME_DEPTH:
            moves = root_board.get_legal_moves(rollout_player.get_color())
            if not moves:
                root_board.pass_move()
                rollout_player = rollout_player.opponent
                moves_made += 1
                continue

            move = semi_random_move(root_board, moves, rollout_player.get_color())
            root_board.place_move(move.get_position(), rollout_player.get_color())

            rollout_player = rollout_player.opponent
            depth += 1
            moves_made += 1

        # 4) Back-propagation
        black_score, white_score = root_board.calculate_score()
        while node is not None:
            node.visits += 1
            # From the root player's perspective
            root_won = (
                (black_score > white_score)
                if root_player.get_color() == BLACK_COLOR
                else (white_score > black_score)
            )
            node.total_wins += int(root_won)
            node = node.parent  # type: ignore

        # 5) Restore the board
        for _ in range(moves_made):
            root_board.undo()

    if len(root.children) == 0:
        return None
    best: Node = next(iter(root.children.values()))
    for node in root.children.values():
        if node.visits > best.visits:
            best = node
    return best.move_from_parent


if __name__ == "__main__":
    black_player, white_player = (
        Player("Black Player", BLACK_COLOR),
        Player("White Player", WHITE_COLOR),
    )
    black_player.opponent, white_player.opponent = white_player, black_player
    board = Board(9, black_player, white_player)
    color = BLACK_COLOR

    while not board.is_terminate():
        row, col = map(int, input("Enter row and col to play: ").split())
        board.place_move((row, col), color)
        color *= -1
        board.print_ascii_board()

        move = mcts(board, white_player, 100)
        if move is not None:
            board.place_move(move.get_position(), white_player.get_color())
            color *= -1
            board.print_ascii_board()
        else:
            print("No move found")
