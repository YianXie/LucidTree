import time

from lucidtree.constants import BLACK_COLOR, BOARD_SIZE, WHITE_COLOR
from lucidtree.go.board import Board
from lucidtree.go.player import Player
from lucidtree.nn.features import encode_board

test_black_player = Player("Black Tester", BLACK_COLOR)
test_white_player = Player("White Tester", WHITE_COLOR)


def test_board_encoding_performance() -> None:
    """Test that board encoding is reasonably fast."""
    board = Board(BOARD_SIZE, test_black_player, test_white_player)

    # Place some moves
    board.place_move((0, 0), BLACK_COLOR)
    board.place_move((1, 1), WHITE_COLOR)
    board.place_move((2, 2), BLACK_COLOR)

    # Encode the board multiple times to measure performance
    start_time = time.time()
    iterations = 1000
    for _ in range(iterations):
        _ = encode_board(board)
    end_time = time.time()

    elapsed = end_time - start_time
    time_per_encoding = elapsed / iterations

    # Should be fast (< 1ms per encoding on modern hardware)
    assert time_per_encoding < 0.01, (
        f"Board encoding too slow: {time_per_encoding * 1000:.3f}ms per encoding"
    )


def test_legal_moves_performance() -> None:
    """Test that getting legal moves is reasonably fast."""
    board = Board(BOARD_SIZE, test_black_player, test_white_player)

    # Place some moves to make the board more complex
    board.place_move((0, 0), BLACK_COLOR)
    board.place_move((1, 1), WHITE_COLOR)
    board.place_move((2, 2), BLACK_COLOR)
    board.place_move((3, 3), WHITE_COLOR)

    # Get legal moves multiple times to measure performance
    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        _ = board.get_legal_moves(BLACK_COLOR)
    end_time = time.time()

    elapsed = end_time - start_time
    time_per_call = elapsed / iterations

    # Should be fast (< 100ms per call for 19x19 board)
    assert time_per_call < 0.1, (
        f"get_legal_moves too slow: {time_per_call * 1000:.3f}ms per call"
    )


def test_place_move_performance() -> None:
    """Test that placing moves is reasonably fast."""
    board = Board(BOARD_SIZE, test_black_player, test_white_player)

    # Place and undo moves multiple times to measure performance
    start_time = time.time()
    iterations = 1000
    for i in range(iterations):
        row = i % (BOARD_SIZE - 1)
        col = (i // BOARD_SIZE) % (BOARD_SIZE - 1)
        board.place_move((row, col), BLACK_COLOR)
        board.undo()
    end_time = time.time()

    elapsed = end_time - start_time
    time_per_operation = elapsed / iterations

    # Should be fast (< 1ms per place+undo operation)
    assert time_per_operation < 0.01, (
        f"place_move+undo too slow: {time_per_operation * 1000:.3f}ms per operation"
    )


def test_calculate_score_performance() -> None:
    """Test that score calculation is reasonably fast."""
    board = Board(BOARD_SIZE, test_black_player, test_white_player)

    # Place some moves to create a more complex board
    board.place_move((0, 0), BLACK_COLOR)
    board.place_move((0, 1), BLACK_COLOR)
    board.place_move((1, 0), BLACK_COLOR)
    board.place_move((8, 8), WHITE_COLOR)
    board.place_move((8, 7), WHITE_COLOR)
    board.place_move((7, 8), WHITE_COLOR)

    # Calculate score multiple times to measure performance
    start_time = time.time()
    iterations = 1000
    for _ in range(iterations):
        _ = board.calculate_score()
    end_time = time.time()

    elapsed = end_time - start_time
    time_per_calculation = elapsed / iterations

    # Should be fast (< 20ms per calculation for 19x19 board)
    assert time_per_calculation < 0.02, (
        f"calculate_score too slow: {time_per_calculation * 1000:.3f}ms per calculation"
    )
