# fmt: off

import time
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
    *,
    use_alpha_beta: bool = True,
    komi: float = KOMI,
    rules: str = RULES,
    deadline: float | None = None,
) -> float:
    if deadline is not None and time.perf_counter() >= deadline:
        return evaluate(board, komi=komi, rules=rules)

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
                komi=komi,
                rules=rules,
                deadline=deadline,
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
                komi=komi,
                rules=rules,
                deadline=deadline,
            )

            board.undo()

            best = min(best, score)
            beta = min(beta, best)
            if use_alpha_beta and beta <= alpha:
                break
        return best


def _root_search_one_depth(
    board: Board,
    isMax: bool,
    search_depth: int,
    *,
    use_alpha_beta: bool,
    komi: float,
    rules: str,
    deadline: float | None,
) -> tuple[tuple[int, int] | None, bool]:
    """
    Evaluate all root moves at a fixed search depth.

    Returns:
        (best_position, completed): completed is False if deadline was hit before
        finishing all root moves (caller should discard this iteration).
    """
    best_score = -INFINITY if isMax else INFINITY
    best_move: Move | None = None

    player = max_player if isMax else min_player
    color = player.get_color()
    moves = _legal_moves_including_pass(board, color)

    for move in moves:
        if deadline is not None and time.perf_counter() >= deadline:
            return (None, False)

        _apply_move(board, move, color)
        consecutive_passes = 1 if move is None else 0

        score = minimax(
            board,
            search_depth - 1,
            not isMax,
            -INFINITY,
            INFINITY,
            consecutive_passes,
            use_alpha_beta=use_alpha_beta,
            komi=komi,
            rules=rules,
            deadline=deadline,
        )

        board.undo()

        if (isMax and score > best_score) or (not isMax and score < best_score):
            best_score = score
            best_move = move

    pos = PASS_MOVE_POSITION if best_move is None else best_move.get_position()
    return (pos, True)


def next_best_move(
    board: Board,
    isMax: bool,
    depth: int = 2,
    *,
    use_alpha_beta: bool = True,
    komi: float = KOMI,
    rules: str = RULES,
    deadline: float | None = None,
    stats_out: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """
    Returns the best placement position

    Args:
        board (Board): the board
        isMax (bool): if the player is maximizing
        depth (int, optional): the depth of the search. Defaults to 2.
        use_alpha_beta (bool): enable alpha-beta pruning
        komi (float): komi for evaluation
        rules (str): rules for evaluation
        deadline (float | None): perf_counter() deadline; None means no time limit
        stats_out (dict | None): if set, receives ``search_depth_reached`` when a
            time limit is used (depth completed for the returned move).

    Returns:
        tuple[int, int]: the best placement position
    """
    if deadline is None:
        result, ok = _root_search_one_depth(
            board,
            isMax,
            depth,
            use_alpha_beta=use_alpha_beta,
            komi=komi,
            rules=rules,
            deadline=None,
        )
        assert ok and result is not None  # nosec
        if stats_out is not None:
            stats_out["search_depth_reached"] = depth
        return result

    best_pos: tuple[int, int] | None = None
    reached = 0

    for d in range(1, depth + 1):
        if time.perf_counter() >= deadline:
            break
        pos, completed = _root_search_one_depth(
            board,
            isMax,
            d,
            use_alpha_beta=use_alpha_beta,
            komi=komi,
            rules=rules,
            deadline=deadline,
        )
        if completed and pos is not None:
            best_pos = pos
            reached = d
        else:
            break

    if best_pos is not None:
        if stats_out is not None:
            stats_out["search_depth_reached"] = reached
        return best_pos

    if stats_out is not None:
        stats_out["search_depth_reached"] = 0

    player = max_player if isMax else min_player
    color = player.get_color()
    moves = _legal_moves_including_pass(board, color)
    if not moves:
        return PASS_MOVE_POSITION
    fallback = moves[0]
    return PASS_MOVE_POSITION if fallback is None else fallback.get_position()
