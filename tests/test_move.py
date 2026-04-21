import pytest

from lucidtree.constants import BLACK_COLOR, EMPTY_COLOR, WHITE_COLOR
from lucidtree.go.exceptions import InvalidColorError
from lucidtree.go.move import Move


class TestMoveInit:
    def test_default_move_is_empty(self) -> None:
        move = Move()
        assert move.is_empty()
        assert not move.is_passed()
        assert move.get_position() == (-1, -1)
        assert move.get_color() == EMPTY_COLOR

    def test_move_with_position_and_color(self) -> None:
        move = Move(3, 5, BLACK_COLOR)
        assert move.get_position() == (3, 5)
        assert move.get_color() == BLACK_COLOR
        assert not move.is_empty()

    def test_pass_move(self) -> None:
        move = Move(passed=True)
        assert move.is_passed()

    def test_non_pass_move(self) -> None:
        move = Move(3, 5, BLACK_COLOR)
        assert not move.is_passed()


class TestMoveSetColor:
    def test_set_valid_black_color(self) -> None:
        move = Move(0, 0, EMPTY_COLOR)
        move.set_color(BLACK_COLOR)
        assert move.get_color() == BLACK_COLOR

    def test_set_valid_white_color(self) -> None:
        move = Move(0, 0, EMPTY_COLOR)
        move.set_color(WHITE_COLOR)
        assert move.get_color() == WHITE_COLOR

    def test_set_empty_color(self) -> None:
        move = Move(0, 0, BLACK_COLOR)
        move.set_color(EMPTY_COLOR)
        assert move.is_empty()

    def test_set_invalid_color_raises(self) -> None:
        move = Move(0, 0, EMPTY_COLOR)
        with pytest.raises(InvalidColorError):
            move.set_color(99)

    def test_set_invalid_color_string_raises(self) -> None:
        move = Move(0, 0, EMPTY_COLOR)
        with pytest.raises(InvalidColorError):
            move.set_color("black")  # type: ignore


class TestMoveEquality:
    def test_equal_moves_same_position(self) -> None:
        m1 = Move(2, 3, BLACK_COLOR)
        m2 = Move(2, 3, WHITE_COLOR)  # color doesn't affect equality
        assert m1 == m2

    def test_equal_moves_same_pass(self) -> None:
        m1 = Move(passed=True)
        m2 = Move(passed=True)
        assert m1 == m2

    def test_unequal_moves_different_row(self) -> None:
        m1 = Move(2, 3, BLACK_COLOR)
        m2 = Move(3, 3, BLACK_COLOR)
        assert m1 != m2

    def test_unequal_moves_different_col(self) -> None:
        m1 = Move(2, 3, BLACK_COLOR)
        m2 = Move(2, 4, BLACK_COLOR)
        assert m1 != m2

    def test_not_equal_to_non_move(self) -> None:
        move = Move(2, 3, BLACK_COLOR)
        assert move.__eq__("not a move") is NotImplemented


class TestMoveHash:
    def test_equal_moves_have_same_hash(self) -> None:
        m1 = Move(2, 3, BLACK_COLOR)
        m2 = Move(2, 3, WHITE_COLOR)
        assert hash(m1) == hash(m2)

    def test_moves_usable_in_set(self) -> None:
        moves = {
            Move(0, 0, BLACK_COLOR),
            Move(0, 1, BLACK_COLOR),
            Move(0, 0, WHITE_COLOR),
        }
        # (0,0,False) and (0,0,False) are equal → deduplicated
        assert len(moves) == 2

    def test_pass_moves_hashable(self) -> None:
        move = Move(passed=True)
        s = {move}
        assert move in s


class TestMoveLessThan:
    def test_smaller_row_is_less(self) -> None:
        m1 = Move(1, 5)
        m2 = Move(2, 0)
        assert m1 < m2

    def test_same_row_smaller_col_is_less(self) -> None:
        m1 = Move(3, 2)
        m2 = Move(3, 5)
        assert m1 < m2

    def test_not_less_than_non_move_returns_not_implemented(self) -> None:
        move = Move(2, 3)
        assert move.__lt__("other") is NotImplemented

    def test_moves_sortable(self) -> None:
        moves = [Move(3, 3), Move(0, 0), Move(1, 5)]
        sorted_moves = sorted(moves)
        assert sorted_moves[0].get_position() == (0, 0)
        assert sorted_moves[1].get_position() == (1, 5)
        assert sorted_moves[2].get_position() == (3, 3)


class TestMoveRepr:
    def test_repr_contains_position_and_color(self) -> None:
        move = Move(2, 3, BLACK_COLOR)
        r = repr(move)
        assert "2" in r
        assert "3" in r
        assert str(BLACK_COLOR) in r
