"""
Tests for the position plumbing in the collection harness.

The pool itself is not exercised here -- spawning processes and loading a
checkpoint per worker does not belong in the unit suite -- but everything that
decides *what* gets searched does, since a silently malformed position would
corrupt a collection run without failing it.
"""

# fmt: off
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.collect_positions import (PASS, SPREAD_POINTS, build_position,
                                       default_positions, load_positions)

# fmt: on


def test_spread_points_are_pairwise_non_adjacent() -> None:
    """Every prefix must be legal, which requires no two points touching."""
    points = set(SPREAD_POINTS)
    assert len(points) == len(SPREAD_POINTS)

    for row, col in points:
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            assert (row + d_row, col + d_col) not in points


@pytest.mark.parametrize("count", [1, 8, 24])
def test_default_positions_are_legal(count: int) -> None:
    """Building each default position must not raise an illegal move."""
    positions = default_positions(count)
    assert len(positions) == count

    for _, moves in positions:
        board, to_play = build_position(moves)
        assert to_play.opponent is not None
        assert not board.is_terminate()
        assert len(board.get_legal_moves(to_play.get_color())) == 362 - len(moves)


def test_build_position_alternates_colors() -> None:
    """Moves are replayed by whoever is to move, so colors must alternate."""
    board, to_play = build_position([(3, 3), (15, 15), (9, 9)])

    assert board.get_move_at_position((3, 3)).get_color() == 1
    assert board.get_move_at_position((15, 15)).get_color() == -1
    assert board.get_move_at_position((9, 9)).get_color() == 1
    assert to_play is board.get_white_player()


def test_build_position_handles_pass() -> None:
    """A pass advances the turn without placing a stone."""
    board, to_play = build_position([(3, 3), PASS])

    assert to_play is board.get_black_player()
    assert board.get_last_move() is not None
    assert board.get_last_move().is_passed()  # type: ignore[union-attr]


def test_load_positions_accepts_both_shapes(tmp_path: Path) -> None:
    """Bare move lists and named objects must both round-trip."""
    path = tmp_path / "positions.json"
    path.write_text(
        json.dumps(
            [
                [[3, 3], [15, 15]],
                {"name": "named", "moves": [[9, 9], PASS]},
            ]
        )
    )

    positions = load_positions(path)

    assert [name for name, _ in positions] == ["pos_0000", "named"]
    assert positions[0][1] == [(3, 3), (15, 15)]
    assert positions[1][1] == [(9, 9), PASS]
