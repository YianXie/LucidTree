"""
Equivalence tests for the two rewritten Board/features hot paths.

``_get_legal_moves_reference`` and ``_encode_board_reference`` are the
pre-optimization implementations, frozen here. Both rewrites are supposed to
be output-identical, not merely equivalent-in-spirit, so these tests replay
thousands of random positions - on small boards, which fill up and therefore
produce the surrounded points, captures, kos, eyes and suicide points that the
optimized paths have to fall back on - and compare the results exactly.

For ``get_legal_moves`` that means the same Move objects in the same order,
with the board left in the same state afterwards. For ``encode_board`` it
means a bit-identical tensor of the same dtype.
"""

# fmt: off
from __future__ import annotations

import random

import pytest
import torch

from lucidtree.constants import (BLACK_COLOR, CHANNEL_SIZE, EMPTY_COLOR,
                                 WHITE_COLOR)
from lucidtree.go.board import Board
from lucidtree.go.exceptions import InvalidColorError
from lucidtree.go.move import Move
from lucidtree.go.player import Player
from lucidtree.nn.features import encode_board

# fmt: on

BOARD_SIZES = [5, 7, 9, 19]


def _get_legal_moves_reference(board: Board, color: int) -> list[Move]:
    """Original ``Board.get_legal_moves``."""
    moves: list[Move] = []
    for row in board.state:
        for move in row:
            if not move.is_empty():
                continue
            # Test validity by temporarily setting color
            move.set_color(color)
            is_valid = board.move_is_valid(move)
            move.set_color(EMPTY_COLOR)  # Restore to empty
            if is_valid:
                moves.append(move)
    return moves + [Move(passed=True)]


def _encode_board_reference(board: Board) -> torch.Tensor:
    """Original ``features.encode_board``."""
    x = torch.zeros(CHANNEL_SIZE, board.size, board.size, dtype=torch.int16)

    for i in range(board.size):
        for j in range(board.size):
            color = board.state[i][j].get_color()
            if color == BLACK_COLOR:
                x[0, i, j] = 1  # Black
            elif color == WHITE_COLOR:
                x[1, i, j] = 1  # White
            else:
                x[2, i, j] = 1  # Empty

    # Current player
    if board.get_current_player().get_color() == BLACK_COLOR:
        x[3].fill_(1)

    # Last move
    last_move = board.get_last_move()
    if last_move is not None and not last_move.is_passed():
        row, col = last_move.get_position()
        x[4, row, col] = 1

    # Ko point
    ko_position = board.get_ko_point()
    if ko_position is not None:
        x[5, ko_position[0], ko_position[1]] = 1

    return x


def _make_board(size: int) -> Board:
    """A fresh board of the given size with linked players."""
    black = Player.black()
    white = Player.white()
    black.opponent = white
    white.opponent = black
    return Board(size, black, white)


def _grid(board: Board) -> tuple[tuple[int, ...], ...]:
    """The stone colors, for detecting leaked mutations."""
    return tuple(tuple(m.get_color() for m in row) for row in board.state)


def _compare_legal_moves(board: Board, color: int) -> None:
    """Both implementations must agree, and neither may disturb the board."""
    before = _grid(board)

    want = _get_legal_moves_reference(board, color)
    assert _grid(board) == before

    got = board.get_legal_moves(color)
    assert _grid(board) == before

    assert [m.get_position() for m in got] == [m.get_position() for m in want]
    assert [m.is_passed() for m in got] == [m.is_passed() for m in want]
    # The placement entries must be the board's own Move objects, restored to
    # empty, exactly as the original returned them.
    for move in got[:-1]:
        assert move.get_color() == EMPTY_COLOR
        assert move is board.get_move_at_position(move.get_position())


@pytest.mark.parametrize("size", BOARD_SIZES)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_playouts_agree(size: int, seed: int) -> None:
    """Play random games and compare both hot paths at every position."""
    rng = random.Random(seed)
    board = _make_board(size)

    captures_seen = 0
    for _ in range(size * size * 2):
        color = board.get_current_player().get_color()

        _compare_legal_moves(board, color)
        assert torch.equal(encode_board(board), _encode_board_reference(board))
        assert encode_board(board).dtype == _encode_board_reference(board).dtype

        legal = [m for m in board.get_legal_moves(color) if not m.is_passed()]
        if not legal:
            break

        before_captures = (
            board.get_black_player().get_capture_count()
            + board.get_white_player().get_capture_count()
        )
        board.place_move(rng.choice(legal).get_position(), color)
        after_captures = (
            board.get_black_player().get_capture_count()
            + board.get_white_player().get_capture_count()
        )
        captures_seen += after_captures - before_captures

    # A playout that never captured would not have exercised the fallback.
    # Ko is too rare to hit reliably from random play, so it gets its own
    # deterministic test below rather than a flaky assertion here.
    if size <= 9:
        assert captures_seen > 0, "playout never captured; coverage too weak"


@pytest.mark.parametrize("size", [5, 7])
def test_dense_board_agrees(size: int) -> None:
    """Fill the board almost completely, where the fast path stops applying."""
    rng = random.Random(99)
    board = _make_board(size)

    for _ in range(size * size * 4):
        color = board.get_current_player().get_color()
        legal = [m for m in board.get_legal_moves(color) if not m.is_passed()]
        if not legal:
            break
        board.place_move(rng.choice(legal).get_position(), color)

    empties = sum(row.count(EMPTY_COLOR) for row in _grid(board))
    surrounded = 0
    for r, row in enumerate(board.state):
        for c, move in enumerate(row):
            if move.get_color() != EMPTY_COLOR:
                continue
            neighbors = [
                board.state[r + dr][c + dc]
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < size and 0 <= c + dc < size
            ]
            if all(n.get_color() != EMPTY_COLOR for n in neighbors):
                surrounded += 1

    assert empties > 0
    assert surrounded > 0, "no fully surrounded point; fallback not exercised"

    for color in (BLACK_COLOR, WHITE_COLOR):
        _compare_legal_moves(board, color)
    assert torch.equal(encode_board(board), _encode_board_reference(board))


def test_ko_point_is_excluded() -> None:
    """The ko point stays illegal, and it reaches the fallback to prove it."""
    board = _make_board(19)
    for move in [
        (8, 9),
        (9, 9),
        (10, 9),
        (8, 10),
        (9, 8),
        (10, 10),
        (3, 3),
        (9, 11),
        (9, 10),
    ]:
        board.place_move(move, board.get_current_player().get_color())

    assert board.get_ko_point() == (9, 9)
    color = board.get_current_player().get_color()
    positions = {m.get_position() for m in board.get_legal_moves(color)}
    assert (9, 9) not in positions
    _compare_legal_moves(board, color)


def test_invalid_color_raises() -> None:
    """An invalid color is rejected, as the original's set_color call was."""
    board = _make_board(9)
    with pytest.raises(InvalidColorError):
        board.get_legal_moves(7)


def test_encode_board_planes_after_pass() -> None:
    """A pass clears the last-move plane in both implementations."""
    board = _make_board(9)
    board.place_move((4, 4), BLACK_COLOR)
    board.pass_move()

    assert torch.equal(encode_board(board), _encode_board_reference(board))
    assert int(encode_board(board)[4].sum()) == 0
