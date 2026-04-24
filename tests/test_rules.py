# fmt: off

from lucidtree.constants import (BLACK_COLOR, BOARD_SIZE, EMPTY_COLOR,
                                 WHITE_COLOR)
from lucidtree.go.rules import Rules

# fmt: on


class TestRowColIsValid:
    def test_origin_is_valid(self) -> None:
        assert Rules.row_col_is_valid(0, 0)

    def test_last_position_is_valid(self) -> None:
        assert Rules.row_col_is_valid(BOARD_SIZE - 1, BOARD_SIZE - 1)

    def test_pass_position_is_valid(self) -> None:
        assert Rules.row_col_is_valid(-1, -1)

    def test_negative_row_is_invalid(self) -> None:
        assert not Rules.row_col_is_valid(-2, 0)

    def test_negative_col_is_invalid(self) -> None:
        assert not Rules.row_col_is_valid(0, -2)

    def test_row_too_large_is_invalid(self) -> None:
        assert not Rules.row_col_is_valid(BOARD_SIZE, 0)

    def test_col_too_large_is_invalid(self) -> None:
        assert not Rules.row_col_is_valid(0, BOARD_SIZE)

    def test_non_int_row_is_invalid(self) -> None:
        assert not Rules.row_col_is_valid("0", 0)  # type: ignore

    def test_non_int_col_is_invalid(self) -> None:
        assert not Rules.row_col_is_valid(0, "0")  # type: ignore

    def test_custom_board_size(self) -> None:
        assert Rules.row_col_is_valid(12, 12, board_size=13)
        assert not Rules.row_col_is_valid(13, 12, board_size=13)


class TestIndexIsValid:
    def test_zero_is_valid(self) -> None:
        assert Rules.index_is_valid(0)

    def test_last_valid_index(self) -> None:
        assert Rules.index_is_valid(BOARD_SIZE * BOARD_SIZE)  # pass index

    def test_negative_index_is_invalid(self) -> None:
        assert not Rules.index_is_valid(-1)

    def test_too_large_index_is_invalid(self) -> None:
        assert not Rules.index_is_valid(BOARD_SIZE * BOARD_SIZE + 1)

    def test_non_int_is_invalid(self) -> None:
        assert not Rules.index_is_valid(0.5)  # type: ignore


class TestGtpMoveIsValid:
    def test_pass_is_valid(self) -> None:
        assert Rules.gtp_move_is_valid("PASS")

    def test_lowercase_pass_is_valid(self) -> None:
        assert Rules.gtp_move_is_valid("pass")

    def test_a1_is_valid(self) -> None:
        assert Rules.gtp_move_is_valid("A1")

    def test_t19_is_valid(self) -> None:
        assert Rules.gtp_move_is_valid("T19")

    def test_j10_is_valid(self) -> None:
        assert Rules.gtp_move_is_valid("J10")

    def test_empty_string_is_invalid(self) -> None:
        assert not Rules.gtp_move_is_valid("")

    def test_digit_column_is_invalid(self) -> None:
        assert not Rules.gtp_move_is_valid("15")

    def test_alpha_row_is_invalid(self) -> None:
        assert not Rules.gtp_move_is_valid("AA")

    def test_i_column_is_invalid(self) -> None:
        # Rules.gtp_move_is_valid also enforces the GTP I-skip convention
        assert not Rules.gtp_move_is_valid("I5")


class TestColorIsValid:
    def test_black_color_is_valid(self) -> None:
        assert Rules.color_is_valid(BLACK_COLOR)

    def test_white_color_is_valid(self) -> None:
        assert Rules.color_is_valid(WHITE_COLOR)

    def test_empty_color_is_valid(self) -> None:
        assert Rules.color_is_valid(EMPTY_COLOR)

    def test_invalid_color_99_is_invalid(self) -> None:
        assert not Rules.color_is_valid(99)

    def test_invalid_color_negative_is_invalid(self) -> None:
        assert not Rules.color_is_valid(-99)

    def test_non_int_is_invalid(self) -> None:
        assert not Rules.color_is_valid("black")  # type: ignore


class TestPlayerNameIsValid:
    def test_string_name_is_valid(self) -> None:
        assert Rules.player_name_is_valid("Alice")

    def test_empty_string_is_valid(self) -> None:
        assert Rules.player_name_is_valid("")

    def test_non_string_is_invalid(self) -> None:
        assert not Rules.player_name_is_valid(42)  # type: ignore

    def test_none_is_invalid(self) -> None:
        assert not Rules.player_name_is_valid(None)  # type: ignore
