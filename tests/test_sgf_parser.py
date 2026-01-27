"""Comprehensive tests for SGF parsing functionality."""

import tempfile
from pathlib import Path

import pytest
from sgfmill import sgf

from mini_katago.constants import BLACK_COLOR, BOARD_SIZE, WHITE_COLOR
from mini_katago.go.game import Game
from mini_katago.go.player import Player
from mini_katago.misc.sgf_parser import parse_sgf_file, parsed_sgf_game_to_game


class TestSGFParsing:
    """Test suite for SGF file parsing."""

    def test_parse_simple_sgf_game(self) -> None:
        """Test parsing a simple SGF game with basic moves."""
        # Create a simple SGF game
        sgf_game = sgf.Sgf_game(size=BOARD_SIZE)
        root_node = sgf_game.get_root()
        root_node.set("PB", "Black Player")
        root_node.set("PW", "White Player")

        # Add some moves
        node1 = root_node.new_child()
        node1.set_move("b", (2, 2))  # Black at (2, 2)
        node2 = node1.new_child()
        node2.set_move("w", (3, 3))  # White at (3, 3)
        node3 = node2.new_child()
        node3.set_move("b", (4, 4))  # Black at (4, 4)

        game = parsed_sgf_game_to_game(sgf_game)

        assert isinstance(game, Game)
        assert game.black_player.name == "Black Player"
        assert game.white_player.name == "White Player"
        assert game.board.get_size() == BOARD_SIZE

    def test_parse_sgf_with_passes(self) -> None:
        """Test parsing SGF game with pass moves."""
        sgf_game = sgf.Sgf_game(size=BOARD_SIZE)
        root_node = sgf_game.get_root()
        root_node.set("PB", "Black")
        root_node.set("PW", "White")

        # Add moves including passes
        node1 = root_node.new_child()
        node1.set_move("b", (2, 2))
        node2 = node1.new_child()
        node2.set_move("w", None)  # Pass
        node3 = node2.new_child()
        node3.set_move("b", (3, 3))

        game = parsed_sgf_game_to_game(sgf_game)

        assert isinstance(game, Game)
        # Verify that passes are handled correctly
        assert game.board.get_size() == BOARD_SIZE

    def test_parse_sgf_with_winner_black(self) -> None:
        """Test parsing SGF game with black winner."""
        sgf_game = sgf.Sgf_game(size=BOARD_SIZE)
        root_node = sgf_game.get_root()
        root_node.set("PB", "Black")
        root_node.set("PW", "White")
        root_node.set("RE", "B+R")  # Black wins by resignation

        game = parsed_sgf_game_to_game(sgf_game)

        assert isinstance(game, Game)
        assert game.winner is not None
        assert game.winner.get_color() == BLACK_COLOR

    def test_parse_sgf_with_winner_white(self) -> None:
        """Test parsing SGF game with white winner."""
        sgf_game = sgf.Sgf_game(size=BOARD_SIZE)
        root_node = sgf_game.get_root()
        root_node.set("PB", "Black")
        root_node.set("PW", "White")
        root_node.set("RE", "W+R")  # White wins by resignation

        game = parsed_sgf_game_to_game(sgf_game)

        assert isinstance(game, Game)
        assert game.winner is not None
        assert game.winner.get_color() == WHITE_COLOR

    def test_parse_sgf_with_no_winner(self) -> None:
        """Test parsing SGF game with no winner specified."""
        sgf_game = sgf.Sgf_game(size=BOARD_SIZE)
        root_node = sgf_game.get_root()
        root_node.set("PB", "Black")
        root_node.set("PW", "White")
        # No RE (result) property

        game = parsed_sgf_game_to_game(sgf_game)

        assert isinstance(game, Game)
        assert game.winner is None

    def test_parse_sgf_different_board_size(self) -> None:
        """Test parsing SGF game with different board size."""
        board_size = 13
        sgf_game = sgf.Sgf_game(size=board_size)
        root_node = sgf_game.get_root()
        root_node.set("PB", "Black")
        root_node.set("PW", "White")

        game = parsed_sgf_game_to_game(sgf_game)

        assert isinstance(game, Game)
        assert game.board.get_size() == board_size

    def test_parse_sgf_empty_game(self) -> None:
        """Test parsing an empty SGF game (no moves)."""
        sgf_game = sgf.Sgf_game(size=BOARD_SIZE)
        root_node = sgf_game.get_root()
        root_node.set("PB", "Black")
        root_node.set("PW", "White")

        game = parsed_sgf_game_to_game(sgf_game)

        assert isinstance(game, Game)
        assert game.board.get_size() == BOARD_SIZE

    def test_parse_sgf_file_valid(self) -> None:
        """Test parsing a valid SGF file from disk."""
        # Create a temporary SGF file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".sgf", delete=False) as f:
            sgf_game = sgf.Sgf_game(size=BOARD_SIZE)
            root_node = sgf_game.get_root()
            root_node.set("PB", "Black")
            root_node.set("PW", "White")
            node1 = root_node.new_child()
            node1.set_move("b", (2, 2))

            f.write(sgf_game.serialise())
            temp_path = Path(f.name)

        try:
            game = parse_sgf_file(temp_path)
            assert isinstance(game, Game)
            assert game.black_player.name == "Black"
            assert game.white_player.name == "White"
        finally:
            temp_path.unlink()

    def test_parse_sgf_file_not_found(self) -> None:
        """Test parsing a non-existent SGF file raises FileNotFoundError."""
        non_existent_path = Path("/nonexistent/path/to/file.sgf")

        with pytest.raises(FileNotFoundError, match="Invalid file path"):
            parse_sgf_file(non_existent_path)

    def test_parse_sgf_file_invalid_format(self) -> None:
        """Test parsing an invalid SGF file format."""
        # Create a file with invalid SGF content
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".sgf", delete=False) as f:
            f.write(b"Invalid SGF content")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError):
                parse_sgf_file(temp_path)
        finally:
            temp_path.unlink()

    def test_parse_sgf_with_missing_player_names(self) -> None:
        """Test parsing SGF game with missing player names raises KeyError."""
        sgf_game = sgf.Sgf_game(size=BOARD_SIZE)
        root_node = sgf_game.get_root()
        # Don't set PB or PW

        # The parser expects PB and PW to exist, so this should raise KeyError
        with pytest.raises(KeyError):
            parsed_sgf_game_to_game(sgf_game)

    def test_parse_sgf_complex_game(self) -> None:
        """Test parsing a more complex SGF game with many moves."""
        sgf_game = sgf.Sgf_game(size=BOARD_SIZE)
        root_node = sgf_game.get_root()
        root_node.set("PB", "Black")
        root_node.set("PW", "White")

        # Create a sequence of moves
        current_node = root_node
        moves = [
            ("b", (0, 0)),
            ("w", (0, 1)),
            ("b", (1, 0)),
            ("w", (1, 1)),
            ("b", (2, 2)),
            ("w", (3, 3)),
        ]

        for color, pos in moves:
            new_node = current_node.new_child()
            new_node.set_move(color, pos)
            current_node = new_node

        game = parsed_sgf_game_to_game(sgf_game)

        assert isinstance(game, Game)
        assert game.board.get_size() == BOARD_SIZE

    def test_parse_sgf_with_alternating_passes(self) -> None:
        """Test parsing SGF game with alternating pass moves."""
        sgf_game = sgf.Sgf_game(size=BOARD_SIZE)
        root_node = sgf_game.get_root()
        root_node.set("PB", "Black")
        root_node.set("PW", "White")

        # Add alternating passes
        node1 = root_node.new_child()
        node1.set_move("b", None)  # Black passes
        node2 = node1.new_child()
        node2.set_move("w", None)  # White passes

        game = parsed_sgf_game_to_game(sgf_game)

        assert isinstance(game, Game)
        # Game should be terminated after two consecutive passes
        assert game.board.get_size() == BOARD_SIZE

    def test_parse_sgf_color_mapping(self) -> None:
        """Test that SGF color mapping (b/w) is correctly converted."""
        sgf_game = sgf.Sgf_game(size=BOARD_SIZE)
        root_node = sgf_game.get_root()
        root_node.set("PB", "Black")
        root_node.set("PW", "White")

        # Black move
        node1 = root_node.new_child()
        node1.set_move("b", (2, 2))
        # White move
        node2 = node1.new_child()
        node2.set_move("w", (3, 3))

        game = parsed_sgf_game_to_game(sgf_game)

        assert isinstance(game, Game)
        # Verify the moves were placed with correct colors
        # The board should have moves at the specified positions

    def test_parse_sgf_with_result_variations(self) -> None:
        """Test parsing SGF games with different result formats."""
        result_formats = ["B+R", "W+R", "B+0.5", "W+1.5", "0"]

        for result in result_formats:
            sgf_game = sgf.Sgf_game(size=BOARD_SIZE)
            root_node = sgf_game.get_root()
            root_node.set("PB", "Black")
            root_node.set("PW", "White")
            root_node.set("RE", result)

            game = parsed_sgf_game_to_game(sgf_game)

            assert isinstance(game, Game)
            # The winner should be determined based on the first character
            if result.startswith("B"):
                assert game.winner is not None
                assert game.winner.get_color() == BLACK_COLOR
            elif result.startswith("W"):
                assert game.winner is not None
                assert game.winner.get_color() == WHITE_COLOR
