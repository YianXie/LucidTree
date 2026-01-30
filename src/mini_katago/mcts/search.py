# mini_katago/mcts/search.py

"""
A combination of Monte Carlo Tree Search and Neural Network (rollout is still heuristic).

Key changes vs your current file:
- Remove deepcopy per simulation (use board.place_move + board.undo along the simulation path)
- Always include PASS as legal move
- Fix value perspective: value is ALWAYS from root_player perspective
- Expand one move per visit (incremental expansion), not full expansion
- Return a concrete best move from root (by visits)
"""

# fmt: off

from typing import Optional, Tuple

from mini_katago import utils
from mini_katago.constants import (ADJ_BOOST, BLACK_COLOR, BOARD_SIZE,
                                   CAPTURE_BOOST, MAX_GAME_DEPTH,
                                   NUM_SIMULATIONS, PASS_MOVE_POSITION,
                                   WHITE_COLOR)
from mini_katago.go.board import Board
from mini_katago.go.move import Move
from mini_katago.go.player import Player
from mini_katago.mcts.node import Node

# fmt: on


def move_weight(
    board: Board,
    move: Move,
    color: int,
    *,
    capture_boost: float = CAPTURE_BOOST,
    adj_boost: float = ADJ_BOOST,
) -> float:
    """
    Weight a move based on its capture count and whether it is connected to own stones.

    NOTE: Your original capture_boost exponent is very aggressive; leaving it as-is
    to preserve your idea, but it's usually too spiky for rollouts.
    """
    if move.is_passed():
        return 1.0

    weight = 1.0
    prev_color = move.get_color()
    move.set_color(color)

    captures = board.check_captures(move)
    if captures:
        weight *= len(captures) ** capture_boost

    neighbors = board.get_neighbors(move)
    for neighbor in neighbors:  # type: ignore
        if neighbor.get_color() == move.get_color():
            weight *= adj_boost
            break

    move.set_color(prev_color)
    return weight


def semi_random_move(board: Board, legal_moves: list[Move], color: int) -> Move:
    weights = [move_weight(board, move, color) for move in legal_moves]
    choice = utils.weighted_choice(legal_moves, weights)
    assert isinstance(choice, Move)
    return choice


def _apply_pos(board: Board, pos: Tuple[int, int], color: int) -> None:
    if pos == PASS_MOVE_POSITION:
        board.pass_move()
    else:
        board.place_move(pos, color)


def _winner_value_from_root(board: Board, root_color: int) -> float:
    """
    +1 if root_color wins, 0 draw, -1 loss.
    """
    black_score, white_score = board.calculate_score()
    if black_score == white_score:
        return 0.0

    winner_color = BLACK_COLOR if black_score > white_score else WHITE_COLOR
    return 1.0 if winner_color == root_color else -1.0


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
        Returns the root Node; use best_move_from_root() to get an actual move.
        """
        root = Node(
            prior=1.0,
            player_to_play=root_player,
            parent=None,
            move_from_parent=None,
        )
        root.expand(root_board)

        root_color = root_player.get_color()

        for _ in range(num_simulations):
            node = root
            player = root_player

            # We will apply moves directly to root_board and then undo them.
            moves_applied = 0
            search_path: list[Node] = [node]

            # 1) Selection: walk down while fully expanded and has children
            while node.is_expanded and node.fully_expanded and node.children:
                pos, node = node.select_child()
                _apply_pos(root_board, pos, player.get_color())
                moves_applied += 1
                search_path.append(node)
                player = player.opponent

            # 2) Expansion: expand ONE child (if possible)
            if not root_board.is_terminate():
                child = node.expand_one(root_board)
                if child is not None:
                    # apply the expanded move (derive position from the parent's children mapping)
                    # find which pos maps to this child
                    expanded_pos: Optional[Tuple[int, int]] = None
                    for pos, (_, ch) in node.children.items():
                        if ch is child:
                            expanded_pos = pos
                            break

                    if expanded_pos is not None:
                        _apply_pos(root_board, expanded_pos, player.get_color())
                        moves_applied += 1
                        node = child
                        search_path.append(node)
                        player = player.opponent

            # 3) Rollout (temporary heuristic policy)
            depth = 0
            rollout_player = player
            while not root_board.is_terminate() and depth < MAX_GAME_DEPTH:
                depth += 1
                legal_moves = (
                    root_board.get_legal_moves(rollout_player.get_color()) or []
                )

                # Filter out pass moves to check if there are any placement moves
                placement_moves = [move for move in legal_moves if not move.is_passed()]
                if not placement_moves:
                    # No placement moves available, must pass
                    root_board.pass_move()
                    moves_applied += 1
                    rollout_player = rollout_player.opponent
                    continue

                move = semi_random_move(
                    root_board, legal_moves, rollout_player.get_color()
                )
                if move.is_passed():
                    root_board.pass_move()
                else:
                    root_board.place_move(
                        move.get_position(), rollout_player.get_color()
                    )
                moves_applied += 1
                rollout_player = rollout_player.opponent

            # 4) Back-propagate value (root perspective, then flip each level)
            value = _winner_value_from_root(root_board, root_color)
            self._backup(search_path, value)

            # 5) Undo all moves from this simulation
            for _ in range(moves_applied):
                root_board.undo()

        return root

    def _backup(self, search_path: list[Node], value: float) -> None:
        """
        value is from ROOT player's perspective.
        Flip sign as we go up because players alternate.
        """
        v = value
        for node in reversed(search_path):
            node.visits += 1
            node.total_value += v
            v = -v

    def best_move_from_root(self, root: Node) -> Tuple[int, int]:
        """
        Choose the best move at the root by max child visits.
        Returns (row,col) or PASS_POS.
        """
        if not root.children:
            return PASS_MOVE_POSITION
        best_pos, (_, best_child) = max(
            root.children.items(), key=lambda kv: kv[1][1].visits
        )
        return best_pos

    def best_move(
        self,
        board: Board,
        player: Player,
        num_simulations: int = NUM_SIMULATIONS,
    ) -> Tuple[int, int]:
        """
        Convenience: run MCTS and return best move position.
        """
        root = self.run(board, player, num_simulations=num_simulations)
        return self.best_move_from_root(root)


# Demo
if __name__ == "__main__":
    mcts = MCTS()
    black_player, white_player = (
        Player("Black Player", BLACK_COLOR),
        Player("White Player", WHITE_COLOR),
    )
    black_player.opponent, white_player.opponent = white_player, black_player
    board = Board(BOARD_SIZE, black_player, white_player)

    while not board.is_terminate():
        row, col = map(
            int, input("Enter a position to play (row col), or -1 -1 to pass: ").split()
        )

        # Human (black)
        if (row, col) == PASS_MOVE_POSITION:
            board.pass_move()
        else:
            board.place_move((row, col), BLACK_COLOR)

        board.print_ascii_board()

        # AI (white)
        best = mcts.best_move(board, white_player, num_simulations=200)
        if best == PASS_MOVE_POSITION:
            board.pass_move()
            print("AI plays: PASS")
        else:
            board.place_move(best, WHITE_COLOR)
            print(f"AI plays: {best}")

        board.print_ascii_board()
