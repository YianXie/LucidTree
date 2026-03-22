# fmt: off

from lucidtree.constants import (BLACK_COLOR, INFINITY, PASS_MOVE_POSITION,
                                 WHITE_COLOR)
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


def evaluate(board: Board) -> float:
    """
    Simple eval:
    - uses board.calculate_score() if present (territory-ish)
    - plus small capture bonus
    positive favors white, negative favors black
    """
    # territory/score estimate
    try:
        black_score, white_score = board.calculate_score()
        score_term = white_score - black_score
    except Exception:
        score_term = 0

    black_captures = board.get_black_player().capture_count
    white_captures = board.get_white_player().capture_count
    capture_term = (white_captures - black_captures) * 0.5  # keep it small

    return float(score_term) + float(capture_term)


def minimax(
    board: Board,
    depth: int,
    isMax: bool,
    alpha: float,
    beta: float,
    consecutive_passes: int,
) -> float:
    """
    Depth-limited minimax with alpha-beta pruning.
    isMax=True => white to play (maximize)
    isMax=False => black to play (minimize)
    """
    if depth <= 0 or game_is_over_by_passes(consecutive_passes) or board.is_terminate():
        return evaluate(board)

    player = max_player if isMax else min_player
    color = player.get_color()

    moves = _legal_moves_including_pass(board, color)

    if isMax:
        best = -INFINITY
        for move in moves:
            # apply
            _apply_move(board, move, color)
            new_passes = consecutive_passes + 1 if move is None else 0

            score = minimax(board, depth - 1, False, alpha, beta, new_passes)

            # undo
            board.undo()

            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best

    else:
        best = INFINITY
        for move in moves:
            _apply_move(board, move, color)
            new_passes = consecutive_passes + 1 if move is None else 0

            score = minimax(board, depth - 1, True, alpha, beta, new_passes)

            board.undo()

            best = min(best, score)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


def next_best_move(board: Board, isMax: bool, depth: int = 2) -> tuple[int, int]:
    """
    Returns the best placement position

    Args:
        board (Board): the board
        isMax (bool): if the player is maximizing
        depth (int, optional): the depth of the search. Defaults to 2.

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
            board, depth - 1, not isMax, -INFINITY, INFINITY, consecutive_passes
        )

        board.undo()

        if (isMax and score > best_score) or (not isMax and score < best_score):
            best_score = score
            best_move = move

    return PASS_MOVE_POSITION if best_move is None else best_move.get_position()
