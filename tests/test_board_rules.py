import pytest

from mini_katago.constants import BLACK_COLOR, BOARD_SIZE, WHITE_COLOR
from mini_katago.go.board import Board
from mini_katago.go.player import Player

test_black_player = Player("Black Tester", BLACK_COLOR)
test_white_player = Player("White Tester", WHITE_COLOR)


def test_single_stone_capture() -> None:
    board = Board(BOARD_SIZE, test_black_player, test_white_player)

    # Place 3 black pieces
    board.place_move((1, 1), BLACK_COLOR)
    board.place_move((2, 0), BLACK_COLOR)
    board.place_move((3, 1), BLACK_COLOR)

    # Place 1 white piece at (2, 1) so it has only 1 liberty
    board.place_move((2, 1), WHITE_COLOR)
    assert board.get_move_at_position((2, 1)).get_color() == WHITE_COLOR

    # Place 1 black piece to capture the white piece
    board.place_move((2, 3), BLACK_COLOR)

    assert board.get_move_at_position((2, 2)).is_empty(), "White should be captured"


def test_group_capture() -> None:
    board = Board(BOARD_SIZE, test_black_player, test_white_player)

    # White group: two stones connected horizontally
    board.place_move((2, 2), WHITE_COLOR)
    board.place_move((2, 3), WHITE_COLOR)

    # Surround the group with black stones
    black_moves = [
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 4),
        (3, 2),
        (3, 3),
    ]
    for black_move in black_moves:
        board.place_move(black_move, BLACK_COLOR)

    assert board.get_move_at_position((2, 2)).is_empty()
    assert board.get_move_at_position((2, 3)).is_empty()


def test_self_suicide_is_illegal() -> None:
    board = Board(BOARD_SIZE, test_black_player, test_white_player)

    # Black stones create an "eye" that white can't place
    board.place_move((1, 1), BLACK_COLOR)
    board.place_move((2, 0), BLACK_COLOR)
    board.place_move((2, 2), BLACK_COLOR)
    board.place_move((3, 1), BLACK_COLOR)

    prev_board = board.state

    # White tries to place_move inside the "eye"
    with pytest.raises(ValueError) as excinfo:
        board.place_move((2, 1), WHITE_COLOR)

    # Verify we got a ValueError
    assert excinfo.type is ValueError

    # Confirm the board is unchanged
    assert prev_board == board.state


def test_self_suicide_is_legal_if_capture() -> None:
    board = Board(BOARD_SIZE, test_black_player, test_white_player)

    white_moves = [
        (1, 1),
        (2, 0),
        (2, 2),
        (2, 3),
        (3, 1),
    ]
    for white_move in white_moves:
        board.place_move(white_move, WHITE_COLOR)

    black_moves = [
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 3),
        (3, 2),
    ]
    for black_move in black_moves:
        board.place_move(black_move, BLACK_COLOR)

    board.place_move((2, 1), BLACK_COLOR)

    assert board.get_move_at_position((2, 1)).get_color() == BLACK_COLOR


def test_simple_ko_prevents_immediate_recap() -> None:
    board = Board(BOARD_SIZE, test_black_player, test_white_player)

    # White at (1,1)
    board.place_move((1, 1), WHITE_COLOR)

    # Black stones around it, leaving one liberty at (1,2)
    board.place_move((0, 1), BLACK_COLOR)
    board.place_move((1, 0), BLACK_COLOR)
    board.place_move((2, 1), BLACK_COLOR)

    # White stones to make the ko shape (supporting recapture)
    board.place_move((0, 2), WHITE_COLOR)
    board.place_move((2, 2), WHITE_COLOR)
    board.place_move((1, 3), WHITE_COLOR)

    # Black plays at (1,2) capturing white (1,1) -> ko created
    board.place_move((1, 2), BLACK_COLOR)
    assert board.get_move_at_position((1, 1)).is_empty()

    # White attempts immediate recapture at (1,1) (should be illegal by simple ko)
    with pytest.raises(ValueError) as excinfo:
        board.place_move((1, 1), WHITE_COLOR)

    # Verify we got a ValueError
    assert excinfo.type is ValueError

    # Confirm the board is unchanged
    assert board.get_move_at_position((1, 1)).is_empty()
