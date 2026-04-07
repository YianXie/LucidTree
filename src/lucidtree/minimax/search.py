# fmt: off

from typing import Any

from lucidtree.constants import (BLACK_COLOR, INFINITY, KOMI,
                                 PASS_MOVE_POSITION, RULES, WHITE_COLOR)
from lucidtree.go.board import Board
from lucidtree.go.move import Move
from lucidtree.go.player import Player

# fmt: on

# Here, the min player is black, and the max player is white (MiniMax)
min_player, max_player = (
    Player("Black Player", BLACK_COLOR),
    Player("White Player", WHITE_COLOR),
)


def _legal_moves_including_pass(board: Board, color: int) -> list[Move | None]:
    """
    Returns list of legal Move plus a PASS sentinel (None).
    Treats None or [] from get_legal_moves as no placements available.
    """
    legal = board.get_legal_moves(color) or []
    # Filter out pass moves since we'll add None separately
    placement_moves = [move for move in legal if not move.is_passed()]
    moves: list[Move | None] = list(placement_moves)

    return moves + [None]


def _apply_move(board: Board, move: Move | None, color: int) -> None:
    """
    Apply a move; None means PASS.
    """
    if move is None or move.is_passed():
        board.pass_move()
    else:
        board.place_move(move.get_position(), color)


def game_is_over_by_passes(consecutive_passes: int) -> bool:
    """Go ends when both players pass consecutively."""
    return consecutive_passes >= 2


def evaluate(board: Board, *, komi: float = KOMI, rules: str = RULES) -> float:
    """
    Evaluate the current board

    Args:
        board (Board): the current board
        komi (float, optional): the komi value to apply when calculating the scores. Defaults to KOMI.
        rules (str, optional): the rules to apply when calculating the scores. Defaults to RULES.

    Returns:
        float: positive if white is winning, zero if the same, and positive if black is winning
    """
    # territory/score estimate
    black_score, white_score = board.calculate_score(komi=komi, rules=rules)
    return white_score - black_score


def minimax(
    board: Board,
    depth: int,
    isMax: bool,
    alpha: float,
    beta: float,
    consecutive_passes: int,
    **kwargs: Any,
) -> float:
    depth = kwargs.get("depth", 2)
    use_alpha_beta = kwargs.get("use_alpha_beta", True)
    komi = kwargs.get("komi", KOMI)
    rules = kwargs.get("rules", RULES)

    if depth <= 0 or game_is_over_by_passes(consecutive_passes) or board.is_terminate():
        return evaluate(board, komi=komi, rules=rules)

    player = max_player if isMax else min_player
    color = player.get_color()

    moves = _legal_moves_including_pass(board, color)

    if isMax:
        best = -INFINITY
        for move in moves:
            # apply
            _apply_move(board, move, color)
            new_passes = consecutive_passes + 1 if move is None else 0

            score = minimax(
                board,
                depth - 1,
                False,
                alpha,
                beta,
                new_passes,
                use_alpha_beta=use_alpha_beta,
            )

            # undo
            board.undo()

            best = max(best, score)
            alpha = max(alpha, best)
            if use_alpha_beta and beta <= alpha:
                break
        return best

    else:
        best = INFINITY
        for move in moves:
            _apply_move(board, move, color)
            new_passes = consecutive_passes + 1 if move is None else 0

            score = minimax(
                board,
                depth - 1,
                True,
                alpha,
                beta,
                new_passes,
                use_alpha_beta=use_alpha_beta,
            )

            board.undo()

            best = min(best, score)
            beta = min(beta, best)
            if use_alpha_beta and beta <= alpha:
                break
        return best


def next_best_move(
    board: Board, isMax: bool, depth: int = 2, **kwargs: Any
) -> tuple[int, int]:
    """
    Returns the best placement position

    Args:
        board (Board): the board
        isMax (bool): if the player is maximizing
        depth (int, optional): the depth of the search. Defaults to 2.
        **kwargs (Any): other additional arguments

    Returns:
        tuple[int, int]: the best placement position
    """
    best_score = -INFINITY if isMax else INFINITY
    best_move: Move | None = None  # None means PASS

    player = max_player if isMax else min_player
    color = player.get_color()

    moves = _legal_moves_including_pass(board, color)
    for move in moves:
        _apply_move(board, move, color)
        consecutive_passes = 1 if move is None else 0

        score = minimax(
            board,
            depth - 1,
            not isMax,
            -INFINITY,
            INFINITY,
            consecutive_passes,
            **kwargs,
        )

        board.undo()

        if (isMax and score > best_score) or (not isMax and score < best_score):
            best_score = score
            best_move = move

    return PASS_MOVE_POSITION if best_move is None else best_move.get_position()
