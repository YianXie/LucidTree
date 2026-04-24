# fmt: off

from lucidtree.constants import (BLACK_COLOR, BOARD_SIZE, CHANNEL_SIZE,
                                 WHITE_COLOR)
from lucidtree.go.board import Board
from lucidtree.go.player import Player
from lucidtree.nn.features import encode_board

_black = Player("Black", BLACK_COLOR)
_white = Player("White", WHITE_COLOR)

# fmt: on


def _board() -> Board:
    return Board(BOARD_SIZE, _black, _white)


class TestEncodeBoardShape:
    def test_output_tensor_shape(self) -> None:
        board = _board()
        x = encode_board(board)
        assert x.shape == (CHANNEL_SIZE, BOARD_SIZE, BOARD_SIZE)

    def test_output_is_integer_tensor(self) -> None:
        board = _board()
        x = encode_board(board)
        assert x.dtype.is_floating_point is False


class TestEncodeBoardChannels:
    def test_empty_board_black_channel_is_zero(self) -> None:
        board = _board()
        x = encode_board(board)
        assert x[0].sum().item() == 0

    def test_empty_board_white_channel_is_zero(self) -> None:
        board = _board()
        x = encode_board(board)
        assert x[1].sum().item() == 0

    def test_empty_board_empty_channel_all_ones(self) -> None:
        board = _board()
        x = encode_board(board)
        assert x[2].sum().item() == BOARD_SIZE * BOARD_SIZE

    def test_black_stone_in_channel_0(self) -> None:
        board = _board()
        board.place_move((3, 3), BLACK_COLOR)
        x = encode_board(board)
        assert x[0, 3, 3].item() == 1
        assert x[1, 3, 3].item() == 0
        assert x[2, 3, 3].item() == 0

    def test_white_stone_in_channel_1(self) -> None:
        board = _board()
        board.place_move((3, 3), BLACK_COLOR)
        board.place_move((4, 4), WHITE_COLOR)
        x = encode_board(board)
        assert x[1, 4, 4].item() == 1
        assert x[0, 4, 4].item() == 0
        assert x[2, 4, 4].item() == 0

    def test_empty_position_in_channel_2(self) -> None:
        board = _board()
        board.place_move((3, 3), BLACK_COLOR)
        x = encode_board(board)
        assert x[2, 0, 0].item() == 1

    def test_current_player_black_sets_channel_3(self) -> None:
        board = _board()
        # Board starts with black to play
        x = encode_board(board)
        assert x[3].sum().item() == BOARD_SIZE * BOARD_SIZE

    def test_current_player_white_channel_3_is_zero(self) -> None:
        board = _board()
        board.place_move((0, 0), BLACK_COLOR)  # black plays → now white's turn
        x = encode_board(board)
        assert x[3].sum().item() == 0

    def test_last_move_in_channel_4(self) -> None:
        board = _board()
        board.place_move((5, 6), BLACK_COLOR)
        x = encode_board(board)
        assert x[4, 5, 6].item() == 1
        # Only one position set
        assert x[4].sum().item() == 1

    def test_no_last_move_channel_4_all_zero(self) -> None:
        board = _board()
        x = encode_board(board)
        assert x[4].sum().item() == 0

    def test_pass_move_not_in_channel_4(self) -> None:
        board = _board()
        board.pass_move()
        x = encode_board(board)
        assert x[4].sum().item() == 0

    def test_ko_point_in_channel_5(self) -> None:
        # Set up a simple ko position
        board = _board()
        # Create ko scenario: white at (1,1), surrounded by black except (1,2)
        # Black stones around white
        board.place_move((0, 1), BLACK_COLOR)
        board.place_move((1, 0), BLACK_COLOR)
        board.place_move((2, 1), BLACK_COLOR)
        # White stones to make ko shape
        board.place_move((0, 2), WHITE_COLOR)
        board.place_move((2, 2), WHITE_COLOR)
        board.place_move((1, 3), WHITE_COLOR)
        board.place_move((1, 1), WHITE_COLOR)
        # Black captures at (1,2) creating ko
        board.place_move((1, 2), BLACK_COLOR)
        x = encode_board(board)
        # Ko point should be at (1,1) where white was captured
        assert x[5, 1, 1].item() == 1
        assert x[5].sum().item() == 1

    def test_no_ko_channel_5_all_zero(self) -> None:
        board = _board()
        board.place_move((3, 3), BLACK_COLOR)
        x = encode_board(board)
        assert x[5].sum().item() == 0
