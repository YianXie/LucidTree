"""
A pure Monte Carlo Tree Search algorithm for Go
"""

import math
import random
from typing import Self

# fmt: off
from mini_katago.board import Board, Move
from mini_katago.constants import (ADJ_BOOST, BLACK_COLOR, CAPTURE_BOOST,
                                   EXPLORATION_CONSTANT, INFINITY,
                                   MAX_GAME_DEPTH, NUM_SIMULATIONS,
                                   WHITE_COLOR)
from mini_katago.player import Player

# fmt: on


class Node:
    """
    A node represents a game state (board position).
    """

    def __init__(
        self,
        visits: int,
        total_wins: int,
        player_to_play: Player,
        parent: Self | None,
        move_from_parent: Move | None,
    ) -> None:
        """
        Initialize a node object

        Args:
            visits (int): the amount of visits that the node has
            total_wins (int): the accumulated wins of this node
            player_to_play (Player): the player that is about to play next
            parent (Self | None): pointer to the previous node, root has None
            move_from_parent (Move | None): the parent move that leads to this node, root has None
        """
        self.visits = visits
        self.total_wins = total_wins
        self.player_to_play = player_to_play
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.untried_moves: list[Move] | None = None
        self.children: dict[Move, Self] = {}

    def uct_score(self, parent_visits: int, C: float = EXPLORATION_CONSTANT) -> float:
        """
        Calculate the UCT (Upper Confidence Bound applied to Trees) score

        Args:
            parent_visits (int): the node's parent's visits
            C (float, optional): the exploration constant, normally between 1.2-2. Defaults to 1.5.

        Returns:
            float: the UCT score
        """
        if self.visits == 0:
            return INFINITY
        return self.total_wins / self.visits + C * math.sqrt(
            math.log(max(1, parent_visits))  # uses max(1, parent_visits) as a safeguard
            / self.visits
        )

    def select_child(self) -> Self | None:
        """
        Return the child with the highest UCT score

        Returns:
            Self | None: the child with the highest UCT score, or None if node has no children
        """
        best_score = -INFINITY
        best_node: Self | None = None
        if self.children:
            for node in self.children.values():
                score = node.uct_score(node.parent.visits)  # type: ignore
                if score > best_score:
                    best_score = score
                    best_node = node
            return best_node
        return None

    def __repr__(self) -> str:
        """
        Return a developer-friendly message for debugging

        Returns:
            str: a developer friendly message
        """
        return f"visits: {self.visits}, total_wins: {self.total_wins}, player_to_player: {self.player_to_play}, parent: {self.parent}, move_from_parent: {self.move_from_parent}"


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
