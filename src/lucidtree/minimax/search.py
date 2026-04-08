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


def _move_to_position(move: Move | None) -> tuple[int, int]:
    return PASS_MOVE_POSITION if move is None else move.get_position()


def _root_search_all_scores_one_depth(
    board: Board,
    isMax: bool,
    search_depth: int,
    *,
    use_alpha_beta: bool,
    komi: float,
    rules: str,
    deadline: float | None,
) -> tuple[list[tuple[tuple[int, int], float]], bool]:
    """
    Score every legal root move at a fixed search depth.

    Returns:
        (scored_moves, completed): ``scored_moves`` pairs each root position with the
        minimax value of replying at that move. ``completed`` is False if the deadline
        was hit before all root moves were scored (caller should discard the iteration).
    """
    scored: list[tuple[tuple[int, int], float]] = []

    player = max_player if isMax else min_player
    color = player.get_color()
    moves = _legal_moves_including_pass(board, color)

    for move in moves:
        if deadline is not None and time.perf_counter() >= deadline:
            return ([], False)

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

        scored.append((_move_to_position(move), score))

    return (scored, True)


def _top_positions_from_scores(
    scored: list[tuple[tuple[int, int], float]],
    isMax: bool,
    include_top_moves: int,
) -> list[tuple[int, int]]:
    """Order root moves by minimax score and take the first ``include_top_moves``."""
    ordered = sorted(scored, key=lambda t: t[1], reverse=isMax)
    return [pos for pos, _ in ordered[:include_top_moves]]


def next_best_moves(
    board: Board,
    isMax: bool,
    depth: int = 2,
    *,
    use_alpha_beta: bool = True,
    komi: float = KOMI,
    rules: str = RULES,
    deadline: float | None = None,
    stats_out: dict[str, Any] | None = None,
    include_top_moves: int = 1,
) -> list[tuple[int, int]]:
    """
    Returns the top placement positions by minimax score at the chosen depth.

    Args:
        board (Board): the board
        isMax (bool): if the player is maximizing
        depth (int, optional): the depth of the search. Defaults to 2.
        use_alpha_beta (bool): enable alpha-beta pruning
        komi (float): komi for evaluation
        rules (str): rules for evaluation
        deadline (float | None): perf_counter() deadline; None means no time limit
        stats_out (dict | None): if set, receives ``search_depth_reached`` when a
            time limit is used (depth completed for the returned move list).
        include_top_moves (int, optional): the number of top moves to include. Defaults to 1.

    Returns:
        list[tuple[int, int]]: the top moves (best first)
    """
    if deadline is None:
        scored, ok = _root_search_all_scores_one_depth(
            board,
            isMax,
            depth,
            use_alpha_beta=use_alpha_beta,
            komi=komi,
            rules=rules,
            deadline=None,
        )
        assert ok and scored  # nosec
        if stats_out is not None:
            stats_out["search_depth_reached"] = depth
        return _top_positions_from_scores(scored, isMax, include_top_moves)

    best_moves: list[tuple[int, int]] | None = None
    reached = 0

    for d in range(1, depth + 1):
        if time.perf_counter() >= deadline:
            break
        scored, completed = _root_search_all_scores_one_depth(
            board,
            isMax,
            d,
            use_alpha_beta=use_alpha_beta,
            komi=komi,
            rules=rules,
            deadline=deadline,
        )
        if completed and scored:
            best_moves = _top_positions_from_scores(scored, isMax, include_top_moves)
            reached = d
        else:
            break

    if best_moves is not None:
        if stats_out is not None:
            stats_out["search_depth_reached"] = reached
        return best_moves

    if stats_out is not None:
        stats_out["search_depth_reached"] = 0

    player = max_player if isMax else min_player
    color = player.get_color()
    moves = _legal_moves_including_pass(board, color)
    if not moves:
        return [PASS_MOVE_POSITION]
    fallback = moves[0]
    return [_move_to_position(fallback)]
