"""
A simple MiniMax algorithm for Go (fixed for pass + terminal conditions)

Fixes:
- Go ends by TWO consecutive passes (not "no legal moves")
- PASS is always legal and is searched like any other move
- Pass moves are undoable in the tree (assumes board.pass_move() is undoable via board.undo())
- Evaluation uses board.calculate_score() if available + captures (still simple)
"""

import math

from mini_katago.constants import BLACK_COLOR, BOARD_SIZE, WHITE_COLOR
from mini_katago.go.board import Board
from mini_katago.go.move import Move
from mini_katago.go.player import Player

INFINITY = math.inf

# Here, the min player is black, and the max player is white (MiniMax)
min_player, max_player = (
    Player("Black Player", BLACK_COLOR),
    Player("White Player", WHITE_COLOR),
)
board = Board(BOARD_SIZE, min_player, max_player)


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


def next_best_move(board: Board, isMax: bool, depth: int = 2) -> Move | None:
    """
    Returns the best placement Move, or None to indicate PASS.
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

    return best_move


if __name__ == "__main__":
    DEPTH = 2  # raise to 3 only if it remains fast enough

    while True:
        row, col = map(
            int, input("Enter a position (row col), or -1 -1 to pass/quit: ").split()
        )
        if row == -1 and col == -1:
            break

        # Human is black
        board.place_move((row, col), min_player.get_color())
        board.print_ascii_board()

        # AI is white
        move = next_best_move(board, isMax=True, depth=DEPTH)
        if move is None or move.is_passed():
            board.pass_move()
            print("AI plays: PASS")
        else:
            board.place_move(move.get_position(), max_player.get_color())
            print(f"AI plays: {move.get_position()}")

        board.print_ascii_board()

    board.show_board()
