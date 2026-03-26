# fmt: off

import pytest

from lucidtree.constants import PASS_INDEX, PASS_MOVE_POSITION
from lucidtree.go.coordinates import (gtp_to_row_col, row_col_to_gtp,
                                      row_col_to_index)
from lucidtree.go.exceptions import InvalidCoordinateError

# fmt: on


class TestMoveToIndex:
    def test_pass_position_returns_pass_index(self) -> None:
        row, col = PASS_MOVE_POSITION
        assert row_col_to_index(row, col) == PASS_INDEX

    def test_origin_maps_to_index_0(self) -> None:
        assert row_col_to_index(0, 0) == 0

    def test_row_column_encoding(self) -> None:
        from lucidtree.constants import BOARD_SIZE

        # (row, col) → row * BOARD_SIZE + col
        assert row_col_to_index(1, 0) == BOARD_SIZE
        assert row_col_to_index(0, 1) == 1
        assert row_col_to_index(2, 3) == 2 * BOARD_SIZE + 3

    def test_last_board_position(self) -> None:
        from lucidtree.constants import BOARD_SIZE

        last_row, last_col = BOARD_SIZE - 1, BOARD_SIZE - 1
        expected = last_row * BOARD_SIZE + last_col
        assert row_col_to_index(last_row, last_col) == expected


# ---------------------------------------------------------------------------
# gtp_to_row_col
# ---------------------------------------------------------------------------


class TestGtpToRowCol:
    def test_pass_returns_pass_position(self) -> None:
        assert gtp_to_row_col("PASS") == PASS_MOVE_POSITION

    def test_pass_case_insensitive(self) -> None:
        assert gtp_to_row_col("pass") == PASS_MOVE_POSITION

    def test_a1_maps_to_row0_col0(self) -> None:
        assert gtp_to_row_col("A1") == (0, 0)

    def test_h1_maps_to_row0_col7(self) -> None:
        # H is the 8th letter (0-indexed: 7); no skip yet
        assert gtp_to_row_col("H1") == (0, 7)

    def test_j1_maps_to_row0_col8(self) -> None:
        # GTP skips 'I', so J is column 8
        assert gtp_to_row_col("J1") == (0, 8)

    def test_k1_maps_to_row0_col9(self) -> None:
        assert gtp_to_row_col("K1") == (0, 9)

    def test_t19_maps_to_row18_col18(self) -> None:
        # T is the 20th letter; skipping I makes it column 18
        assert gtp_to_row_col("T19") == (18, 18)

    def test_row_number_is_one_based(self) -> None:
        # Row 1 in GTP → row index 0
        assert gtp_to_row_col("A1")[0] == 0
        assert gtp_to_row_col("A19")[0] == 18

    def test_letter_i_is_rejected(self) -> None:
        with pytest.raises(InvalidCoordinateError, match="Invalid GTP move"):
            gtp_to_row_col("I5")

    def test_invalid_column_letter_rejected(self) -> None:
        with pytest.raises(InvalidCoordinateError):
            gtp_to_row_col("15")  # digit as column letter

    def test_invalid_row_digit_rejected(self) -> None:
        with pytest.raises(InvalidCoordinateError):
            gtp_to_row_col("AX")  # non-digit row

    def test_whitespace_stripped(self) -> None:
        assert gtp_to_row_col("  A1  ") == (0, 0)

    def test_lowercase_input_accepted(self) -> None:
        assert gtp_to_row_col("d4") == gtp_to_row_col("D4")


# ---------------------------------------------------------------------------
# row_col_to_gtp
# ---------------------------------------------------------------------------


class TestRowColToGtp:
    def test_pass_position_returns_pass(self) -> None:
        assert row_col_to_gtp(-1, -1) == "PASS"

    def test_row0_col0_returns_a1(self) -> None:
        assert row_col_to_gtp(0, 0) == "A1"

    def test_row0_col7_returns_h1(self) -> None:
        assert row_col_to_gtp(0, 7) == "H1"

    def test_row0_col8_returns_j1(self) -> None:
        # Column 8 must skip 'I' and produce 'J'
        assert row_col_to_gtp(0, 8) == "J1"

    def test_row0_col9_returns_k1(self) -> None:
        assert row_col_to_gtp(0, 9) == "K1"

    def test_row18_col18_returns_t19(self) -> None:
        assert row_col_to_gtp(18, 18) == "T19"

    def test_i_never_appears_in_output(self) -> None:
        # No column index should produce the letter 'I'
        for col in range(19):
            gtp = row_col_to_gtp(0, col)
            assert "I" not in gtp, f"'I' found in GTP output for col={col}: {gtp}"


# ---------------------------------------------------------------------------
# Round-trip consistency
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize(
        "gtp_coord",
        ["A1", "B2", "D4", "H8", "J9", "K10", "S19", "T19", "PASS"],
    )
    def test_round_trip(self, gtp_coord: str) -> None:
        row_col = gtp_to_row_col(gtp_coord)
        if row_col == PASS_MOVE_POSITION:
            assert row_col_to_gtp(*row_col) == "PASS"
        else:
            assert row_col_to_gtp(*row_col) == gtp_coord
