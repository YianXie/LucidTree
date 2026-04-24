# fmt: off

import pytest

from lucidtree.constants import BLACK_COLOR, BOARD_SIZE, WHITE_COLOR
from lucidtree.go.board import Board
from lucidtree.go.exceptions import (GameOverError, IllegalMoveError,
                                     InvalidColorError)
from lucidtree.go.move import Move
from lucidtree.go.player import Player

# fmt: on

_black = Player("Black", BLACK_COLOR)
_white = Player("White", WHITE_COLOR)


def _board(size: int = BOARD_SIZE) -> Board:
    return Board(size, _black, _white)


# ---------------------------------------------------------------------------
# pass_move
# ---------------------------------------------------------------------------


class TestPassMove:
    def test_single_pass_switches_player(self) -> None:
        board = _board()
        board.pass_move()
        assert board.get_current_player().get_color() == WHITE_COLOR

    def test_two_consecutive_passes_terminate_game(self) -> None:
        board = _board()
        board.pass_move()
        board.pass_move()
        assert board.is_terminate()

    def test_placing_after_game_over_raises(self) -> None:
        board = _board()
        board.pass_move()
        board.pass_move()
        with pytest.raises(GameOverError):
            board.place_move((0, 0), BLACK_COLOR)

    def test_passing_after_game_over_raises(self) -> None:
        board = _board()
        board.pass_move()
        board.pass_move()
        with pytest.raises(GameOverError):
            board.pass_move()

    def test_one_pass_then_place_resets_consecutive_passes(self) -> None:
        board = _board()
        board.pass_move()
        board.place_move((0, 0), WHITE_COLOR)
        board.pass_move()  # second pass but not consecutive with first
        assert not board.is_terminate()


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------


class TestUndo:
    def test_undo_restores_board_state(self) -> None:
        board = _board()
        board.place_move((3, 3), BLACK_COLOR)
        board.undo()
        assert board.get_move_at_position((3, 3)).is_empty()

    def test_undo_switches_back_player(self) -> None:
        board = _board()
        board.place_move((3, 3), BLACK_COLOR)
        assert board.get_current_player().get_color() == WHITE_COLOR
        board.undo()
        assert board.get_current_player().get_color() == BLACK_COLOR

    def test_undo_pass_switches_back_player(self) -> None:
        board = _board()
        board.pass_move()
        board.undo()
        assert board.get_current_player().get_color() == BLACK_COLOR

    def test_undo_with_no_history_raises(self) -> None:
        board = _board()
        with pytest.raises(RuntimeError):
            board.undo()

    def test_undo_multiple_times(self) -> None:
        board = _board()
        board.place_move((1, 1), BLACK_COLOR)
        board.place_move((2, 2), WHITE_COLOR)
        board.place_move((3, 3), BLACK_COLOR)
        board.undo()
        board.undo()
        board.undo()
        assert board.get_move_at_position((1, 1)).is_empty()
        assert board.get_move_at_position((2, 2)).is_empty()
        assert board.get_move_at_position((3, 3)).is_empty()

    def test_undo_capture_restores_stone(self) -> None:
        """Undoing a capturing move should restore the captured stone."""
        # White at (2,1) has one liberty (2,2); black playing (2,2) captures it
        board = _board()
        board.place_move((1, 1), BLACK_COLOR)
        board.place_move((2, 0), BLACK_COLOR)
        board.place_move((3, 1), BLACK_COLOR)
        board.place_move((2, 1), WHITE_COLOR)
        board.place_move((2, 2), BLACK_COLOR)
        assert board.get_move_at_position((2, 1)).is_empty()
        board.undo()
        assert board.get_move_at_position((2, 1)).get_color() == WHITE_COLOR

    def test_undo_restores_ko_position(self) -> None:
        board = _board()
        # Simple place and undo restores ko to None
        board.place_move((5, 5), BLACK_COLOR)
        board.undo()
        assert board.get_ko_point() is None


# ---------------------------------------------------------------------------
# calculate_score
# ---------------------------------------------------------------------------


class TestCalculateScore:
    def test_empty_board_score_equals_komi(self) -> None:
        board = _board()
        black_score, white_score = board.calculate_score(komi=7.5)
        # Empty board: no territories, white gets komi
        assert black_score == 0.0
        assert white_score == 7.5

    def test_japanese_rules_counts_captures(self) -> None:
        board = _board()
        # Black surrounds and captures 1 white stone
        board.place_move((1, 1), BLACK_COLOR)
        board.place_move((2, 0), BLACK_COLOR)
        board.place_move((3, 1), BLACK_COLOR)
        board.place_move((2, 1), WHITE_COLOR)
        board.place_move((2, 2), BLACK_COLOR)  # captures white at (2,1)
        black_score, _ = board.calculate_score(komi=0, rules="japanese")
        # Black captured 1 stone, so capture_count = 1
        assert black_score >= 1

    def test_chinese_rules_counts_stones(self) -> None:
        board = _board()
        board.place_move((0, 0), BLACK_COLOR)
        black_score_jp, _ = board.calculate_score(komi=0, rules="japanese")
        black_score_cn, _ = board.calculate_score(komi=0, rules="chinese")
        # Chinese counts the stone itself; Japanese counts territory
        assert black_score_cn >= black_score_jp

    def test_invalid_rules_raises(self) -> None:
        board = _board()
        with pytest.raises(ValueError):
            board.calculate_score(rules="american")


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class TestGetLegalMoves:
    def test_empty_board_all_moves_legal_plus_pass(self) -> None:
        board = _board()
        legal = board.get_legal_moves(BLACK_COLOR)
        # All BOARD_SIZE*BOARD_SIZE positions + 1 pass
        assert len(legal) == BOARD_SIZE * BOARD_SIZE + 1

    def test_occupied_position_not_legal(self) -> None:
        board = _board()
        board.place_move((5, 5), BLACK_COLOR)
        legal = board.get_legal_moves(WHITE_COLOR)
        positions = {m.get_position() for m in legal if not m.is_passed()}
        assert (5, 5) not in positions

    def test_pass_always_included(self) -> None:
        board = _board()
        legal = board.get_legal_moves(BLACK_COLOR)
        assert any(m.is_passed() for m in legal)

    def test_suicide_move_not_legal(self) -> None:
        board = _board()
        # Create an eye that would be suicide for white
        board.place_move((1, 1), BLACK_COLOR)
        board.place_move((2, 0), BLACK_COLOR)
        board.place_move((2, 2), BLACK_COLOR)
        board.place_move((3, 1), BLACK_COLOR)
        legal_white = board.get_legal_moves(WHITE_COLOR)
        positions = {m.get_position() for m in legal_white if not m.is_passed()}
        assert (2, 1) not in positions


# ---------------------------------------------------------------------------
# get_nth_move / get_last_move / get_all_moves
# ---------------------------------------------------------------------------


class TestMoveHistory:
    def test_get_last_move_empty_board(self) -> None:
        board = _board()
        assert board.get_last_move() is None

    def test_get_last_move_after_place(self) -> None:
        board = _board()
        board.place_move((3, 3), BLACK_COLOR)
        last = board.get_last_move()
        assert last is not None
        assert last.get_position() == (3, 3)
        assert last.get_color() == BLACK_COLOR

    def test_get_last_move_after_pass(self) -> None:
        board = _board()
        board.pass_move()
        last = board.get_last_move()
        assert last is not None
        assert last.is_passed()

    def test_get_nth_move_positive_index(self) -> None:
        board = _board()
        board.place_move((1, 1), BLACK_COLOR)
        board.place_move((2, 2), WHITE_COLOR)
        first = board.get_nth_move(0)
        assert first is not None
        assert first.get_position() == (1, 1)

    def test_get_nth_move_negative_index(self) -> None:
        board = _board()
        board.place_move((1, 1), BLACK_COLOR)
        board.place_move((2, 2), WHITE_COLOR)
        last = board.get_nth_move(-1)
        assert last is not None
        assert last.get_position() == (2, 2)

    def test_get_nth_move_out_of_range_returns_none(self) -> None:
        board = _board()
        assert board.get_nth_move(0) is None
        board.place_move((1, 1), BLACK_COLOR)
        assert board.get_nth_move(5) is None

    def test_get_all_moves_empty(self) -> None:
        board = _board()
        assert board.get_all_moves() == []

    def test_get_all_moves_tracks_moves(self) -> None:
        board = _board()
        board.place_move((1, 1), BLACK_COLOR)
        board.pass_move()
        board.place_move((3, 3), BLACK_COLOR)
        moves = board.get_all_moves()
        assert len(moves) == 3
        assert moves[0].get_position() == (1, 1)
        assert moves[1].is_passed()
        assert moves[2].get_position() == (3, 3)


# ---------------------------------------------------------------------------
# copy_game_state
# ---------------------------------------------------------------------------


class TestCopyGameState:
    def test_copy_has_same_positions(self) -> None:
        board = _board()
        board.place_move((3, 3), BLACK_COLOR)
        copy = board.copy_game_state()
        assert copy.get_move_at_position((3, 3)).get_color() == BLACK_COLOR

    def test_copy_is_independent(self) -> None:
        board = _board()
        board.place_move((3, 3), BLACK_COLOR)
        copy = board.copy_game_state()
        copy.place_move((4, 4), WHITE_COLOR)
        assert board.get_move_at_position((4, 4)).is_empty()

    def test_copy_has_same_size(self) -> None:
        board = _board(9)
        copy = board.copy_game_state()
        assert copy.get_size() == 9

    def test_copy_has_same_current_player(self) -> None:
        board = _board()
        board.place_move((0, 0), BLACK_COLOR)  # now white's turn
        copy = board.copy_game_state()
        assert copy.get_current_player().get_color() == WHITE_COLOR


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestBoardErrors:
    def test_place_invalid_color_raises(self) -> None:
        board = _board()
        with pytest.raises(InvalidColorError):
            board.place_move((0, 0), 99)

    def test_place_occupied_position_raises(self) -> None:
        board = _board()
        board.place_move((0, 0), BLACK_COLOR)
        with pytest.raises(IllegalMoveError):
            board.place_move((0, 0), WHITE_COLOR)

    def test_place_out_of_bounds_raises(self) -> None:
        from lucidtree.go.exceptions import InvalidCoordinateError

        board = _board()
        with pytest.raises(InvalidCoordinateError):
            board.place_move((BOARD_SIZE, 0), BLACK_COLOR)


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------


class TestGetNeighbors:
    def test_corner_has_two_neighbors(self) -> None:
        board = _board()
        move = board.get_move_at_position((0, 0))
        neighbors = board.get_neighbors(move)
        assert len(neighbors) == 2  # type: ignore

    def test_edge_has_three_neighbors(self) -> None:
        board = _board()
        move = board.get_move_at_position((0, 5))
        neighbors = board.get_neighbors(move)
        assert len(neighbors) == 3  # type: ignore

    def test_center_has_four_neighbors(self) -> None:
        board = _board()
        move = board.get_move_at_position((5, 5))
        neighbors = board.get_neighbors(move)
        assert len(neighbors) == 4  # type: ignore

    def test_pass_move_returns_none(self) -> None:
        board = _board()
        pass_move = Move(passed=True)
        assert board.get_neighbors(pass_move) is None


# ---------------------------------------------------------------------------
# count_liberties
# ---------------------------------------------------------------------------


class TestCountLiberties:
    def test_empty_move_returns_minus_one(self) -> None:
        board = _board()
        move = board.get_move_at_position((5, 5))
        assert board.count_liberties(move) == -1

    def test_isolated_stone_in_center_has_four_liberties(self) -> None:
        board = _board()
        board.place_move((5, 5), BLACK_COLOR)
        move = board.get_move_at_position((5, 5))
        assert board.count_liberties(move) == 4

    def test_corner_stone_has_two_liberties(self) -> None:
        board = _board()
        board.place_move((0, 0), BLACK_COLOR)
        move = board.get_move_at_position((0, 0))
        assert board.count_liberties(move) == 2

    def test_stone_with_neighbor_shares_liberty(self) -> None:
        board = _board()
        board.place_move((5, 5), BLACK_COLOR)
        board.place_move((5, 6), WHITE_COLOR)
        move = board.get_move_at_position((5, 5))
        # (5,5) has neighbors (4,5),(6,5),(5,4),(5,6); (5,6) is white → 3 liberties
        assert board.count_liberties(move) == 3
