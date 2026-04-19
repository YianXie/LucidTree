import pytest

from lucidtree.constants import BLACK_COLOR, WHITE_COLOR
from lucidtree.go.exceptions import InvalidColorError, InvalidNameError
from lucidtree.go.player import Player


class TestPlayerInit:
    def test_player_has_name_and_color(self) -> None:
        p = Player("Alice", BLACK_COLOR)
        assert p.get_name() == "Alice"
        assert p.get_color() == BLACK_COLOR

    def test_initial_capture_count_is_zero(self) -> None:
        p = Player("Bob", WHITE_COLOR)
        assert p.get_capture_count() == 0

    def test_initial_opponent_is_none(self) -> None:
        p = Player("Alice", BLACK_COLOR)
        assert p.opponent is None


class TestPlayerSetName:
    def test_set_valid_name(self) -> None:
        p = Player("Old", BLACK_COLOR)
        p.set_name("New")
        assert p.get_name() == "New"

    def test_set_empty_string_is_valid(self) -> None:
        p = Player("Old", BLACK_COLOR)
        p.set_name("")
        assert p.get_name() == ""

    def test_set_non_string_raises_invalid_name(self) -> None:
        p = Player("Old", BLACK_COLOR)
        with pytest.raises(InvalidNameError):
            p.set_name(123)  # type: ignore


class TestPlayerSetColor:
    def test_set_valid_black_color(self) -> None:
        p = Player("Alice", WHITE_COLOR)
        p.set_color(BLACK_COLOR)
        assert p.get_color() == BLACK_COLOR

    def test_set_valid_white_color(self) -> None:
        p = Player("Alice", BLACK_COLOR)
        p.set_color(WHITE_COLOR)
        assert p.get_color() == WHITE_COLOR

    def test_set_invalid_color_raises(self) -> None:
        p = Player("Alice", BLACK_COLOR)
        with pytest.raises(InvalidColorError):
            p.set_color(42)


class TestPlayerCaptureCount:
    def test_increase_capture_count(self) -> None:
        p = Player("Alice", BLACK_COLOR)
        p.increase_capture_count(3)
        assert p.get_capture_count() == 3

    def test_increase_capture_count_multiple_times(self) -> None:
        p = Player("Alice", BLACK_COLOR)
        p.increase_capture_count(2)
        p.increase_capture_count(5)
        assert p.get_capture_count() == 7

    def test_set_capture_count(self) -> None:
        p = Player("Alice", BLACK_COLOR)
        p.increase_capture_count(10)
        p.set_capture_count(0)
        assert p.get_capture_count() == 0


class TestPlayerStaticMethods:
    def test_black_player(self) -> None:
        p = Player.black()
        assert p.get_color() == BLACK_COLOR
        assert p.get_name() == "Black"

    def test_white_player(self) -> None:
        p = Player.white()
        assert p.get_color() == WHITE_COLOR
        assert p.get_name() == "White"


class TestPlayerRepr:
    def test_repr_contains_name_and_color(self) -> None:
        p = Player("TestName", BLACK_COLOR)
        r = repr(p)
        assert "TestName" in r
        assert str(BLACK_COLOR) in r
