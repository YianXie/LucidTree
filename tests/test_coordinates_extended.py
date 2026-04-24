# fmt: off

import pytest

from lucidtree.constants import BOARD_SIZE, PASS_INDEX, PASS_MOVE_POSITION
from lucidtree.go.coordinates import (gtp_to_index, index_to_gtp,
                                      index_to_row_col, row_col_to_index)
from lucidtree.go.exceptions import InvalidCoordinateError

# fmt: on


class TestIndexToRowCol:
    def test_zero_maps_to_origin(self) -> None:
        assert index_to_row_col(0) == (0, 0)

    def test_pass_index_maps_to_pass_position(self) -> None:
        assert index_to_row_col(PASS_INDEX) == PASS_MOVE_POSITION

    def test_index_1_maps_to_row0_col1(self) -> None:
        assert index_to_row_col(1) == (0, 1)

    def test_board_size_maps_to_row1_col0(self) -> None:
        assert index_to_row_col(BOARD_SIZE) == (1, 0)

    def test_negative_index_raises(self) -> None:
        with pytest.raises(InvalidCoordinateError):
            index_to_row_col(-1)

    def test_too_large_index_raises(self) -> None:
        with pytest.raises(InvalidCoordinateError):
            index_to_row_col(PASS_INDEX + 1)

    def test_round_trip_with_row_col_to_index(self) -> None:
        for row in range(0, BOARD_SIZE, 5):
            for col in range(0, BOARD_SIZE, 5):
                idx = row_col_to_index(row, col)
                assert index_to_row_col(idx) == (row, col)


class TestGtpToIndex:
    def test_a1_to_index_0(self) -> None:
        assert gtp_to_index("A1") == 0

    def test_pass_to_pass_index(self) -> None:
        assert gtp_to_index("PASS") == PASS_INDEX

    def test_j1_to_index(self) -> None:
        # J = column 8 (skips I), row 0 → index 8
        assert gtp_to_index("J1") == 8

    def test_b1_to_index_1(self) -> None:
        assert gtp_to_index("B1") == 1

    def test_invalid_gtp_raises(self) -> None:
        with pytest.raises(InvalidCoordinateError):
            gtp_to_index("I5")


class TestIndexToGtp:
    def test_index_0_to_a1(self) -> None:
        assert index_to_gtp(0) == "A1"

    def test_pass_index_to_pass(self) -> None:
        assert index_to_gtp(PASS_INDEX) == "PASS"

    def test_index_8_to_j1(self) -> None:
        assert index_to_gtp(8) == "J1"

    def test_negative_index_raises(self) -> None:
        with pytest.raises(InvalidCoordinateError):
            index_to_gtp(-1)

    def test_too_large_index_raises(self) -> None:
        with pytest.raises(InvalidCoordinateError):
            index_to_gtp(PASS_INDEX + 1)

    def test_i_never_in_output(self) -> None:
        for idx in range(BOARD_SIZE * BOARD_SIZE):
            gtp = index_to_gtp(idx)
            assert "I" not in gtp, f"'I' found for index {idx}: {gtp}"

    def test_round_trip_gtp_to_index(self) -> None:
        for gtp in ["A1", "B2", "H8", "J9", "T19", "PASS"]:
            assert index_to_gtp(gtp_to_index(gtp)) == gtp
