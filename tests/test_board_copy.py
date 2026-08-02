"""
Isolation and equivalence tests for ``Board.copy``.

``Board.copy`` replaces ``copy.deepcopy`` on the MCTS simulation hot path, so
two things have to hold:

1. **Isolation.** Playing on the copy - including captures, ko and undo, all of
   which reach into the grid, the ko point, the move history and the players'
   capture counts - must leave the original completely untouched. A shallow
   copy that aliased any one of those mutable fields would pass a trivial
   "same stones" check and still corrupt the search.
2. **Equivalence.** The copy must be indistinguishable from ``deepcopy`` of the
   same board: replaying an identical move sequence on both must land on
   identical state.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from lucidtree.constants import BLACK_COLOR, BOARD_SIZE, WHITE_COLOR
from lucidtree.go.board import Board
from lucidtree.go.player import Player

# Canonical ko shape around (9, 9): black's last move captures the lone white
# stone and is itself a lone stone with a single liberty.
KO_SEQUENCE: list[tuple[int, int]] = [
    (8, 9),
    (9, 9),
    (10, 9),
    (8, 10),
    (9, 8),
    (10, 10),
    (3, 3),
    (9, 11),
    (9, 10),
]

# Black surrounds a white stone in the corner and takes it.
CAPTURE_SEQUENCE: list[tuple[int, int]] = [(0, 1), (0, 0), (1, 0)]


def _make_board() -> Board:
    """A fresh 19x19 board with linked players."""
    black = Player.black()
    white = Player.white()
    black.opponent = white
    white.opponent = black
    return Board(BOARD_SIZE, black, white)


def _play(board: Board, moves: list[tuple[int, int]]) -> None:
    """Replay a move sequence using whichever player is to move."""
    for move in moves:
        board.place_move(move, board.get_current_player().get_color())


def _snapshot(board: Board) -> dict[str, Any]:
    """Every externally observable field of a board, as plain data."""
    return {
        "size": board.get_size(),
        "grid": tuple(tuple(m.get_color() for m in row) for row in board.state),
        "ko": board.get_ko_point(),
        "consecutive_passes": board._consecutive_passes,
        "is_terminate": board.is_terminate(),
        "history": [dict(entry) for entry in board._move_history],
        "black_captures": board.get_black_player().get_capture_count(),
        "white_captures": board.get_white_player().get_capture_count(),
        "black_to_play": (board.get_current_player() is board.get_black_player()),
    }


def test_copy_starts_identical() -> None:
    """A fresh copy must be observationally equal to its source."""
    board = _make_board()
    _play(board, KO_SEQUENCE)

    assert _snapshot(board.copy()) == _snapshot(board)


def test_copy_covers_every_field() -> None:
    """A field added to Board later must not be silently dropped by copy()."""
    board = _make_board()
    _play(board, KO_SEQUENCE)

    assert set(board.copy().__dict__) == set(board.__dict__)


def test_copy_does_not_share_players() -> None:
    """The copy needs its own players, or capture counts leak both ways."""
    board = _make_board()
    clone = board.copy()

    assert clone.get_black_player() is not board.get_black_player()
    assert clone.get_white_player() is not board.get_white_player()
    # current_player must alias one of the copy's own players, since
    # place_move switches sides by identity comparison.
    assert clone.get_current_player() is clone.get_black_player()


def test_capture_on_copy_leaves_original_untouched() -> None:
    """A capture mutates the grid and a capture count; neither may leak."""
    board = _make_board()
    _play(board, [(3, 3), (15, 15), (3, 15), (15, 3)])
    before = _snapshot(board)

    clone = board.copy()
    _play(clone, CAPTURE_SEQUENCE)

    assert clone.get_move_at_position((0, 0)).get_color() == 0
    assert clone.get_black_player().get_capture_count() == 1
    assert _snapshot(board) == before


def test_ko_on_copy_leaves_original_untouched() -> None:
    """A ko mutates the ko point as well as the grid."""
    board = _make_board()
    before = _snapshot(board)

    clone = board.copy()
    _play(clone, KO_SEQUENCE)

    assert clone.get_ko_point() == (9, 9)
    assert _snapshot(board) == before
    assert board.get_ko_point() is None


def test_undo_on_copy_leaves_original_untouched() -> None:
    """undo() pops history and restores stones and capture counts."""
    board = _make_board()
    _play(board, KO_SEQUENCE)
    before = _snapshot(board)

    clone = board.copy()
    for _ in range(len(KO_SEQUENCE)):
        clone.undo()

    assert clone._move_history == []
    assert _snapshot(board) == before


def test_pass_on_copy_leaves_original_untouched() -> None:
    """Passing mutates the pass counter and can terminate the game."""
    board = _make_board()
    _play(board, [(3, 3), (15, 15)])
    before = _snapshot(board)

    clone = board.copy()
    clone.pass_move()
    clone.pass_move()

    assert clone.is_terminate()
    assert _snapshot(board) == before
    assert not board.is_terminate()


def test_original_moves_do_not_leak_into_copy() -> None:
    """Isolation has to hold in the other direction too."""
    board = _make_board()
    clone = board.copy()
    clone_before = _snapshot(clone)

    _play(board, KO_SEQUENCE)

    assert _snapshot(clone) == clone_before


@pytest.mark.parametrize(
    "setup,followup",
    [
        ([], KO_SEQUENCE),
        ([(3, 3), (15, 15), (3, 15), (15, 3)], CAPTURE_SEQUENCE),
        (KO_SEQUENCE, [(5, 5), (6, 6), (7, 7)]),
    ],
)
def test_copy_is_equivalent_to_deepcopy(
    setup: list[tuple[int, int]], followup: list[tuple[int, int]]
) -> None:
    """``copy()`` and ``deepcopy`` must be indistinguishable after replay."""
    board = _make_board()
    _play(board, setup)

    fast = board.copy()
    deep = copy.deepcopy(board)

    _play(fast, followup)
    _play(deep, followup)

    assert _snapshot(fast) == _snapshot(deep)


def test_copy_preserves_capture_counts() -> None:
    """Capture counts must carry over, not reset to zero."""
    board = _make_board()
    _play(board, [(3, 3), (15, 15), (3, 15), (15, 3)])
    _play(board, CAPTURE_SEQUENCE)
    assert board.get_black_player().get_capture_count() == 1

    clone = board.copy()
    assert clone.get_black_player().get_capture_count() == 1
    assert clone.get_white_player().get_capture_count() == 0

    # Japanese scoring reads the capture counts, so a reset would silently
    # change every terminal-node evaluation in the search.
    assert clone.calculate_score() == board.calculate_score()


def test_copy_preserves_player_colors_and_link() -> None:
    """The copied players keep their colors and remain each other's opponent."""
    board = _make_board()
    clone = board.copy()

    assert clone.get_black_player().get_color() == BLACK_COLOR
    assert clone.get_white_player().get_color() == WHITE_COLOR
    assert clone.get_black_player().opponent is clone.get_white_player()
    assert clone.get_white_player().opponent is clone.get_black_player()
