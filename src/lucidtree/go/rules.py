# fmt: off
from lucidtree.constants import (BLACK_COLOR, BOARD_SIZE, EMPTY_COLOR, PASS_MOVE_POSITION,
                                 WHITE_COLOR)

# fmt: on

# Pre-compute valid colors set for faster lookup
_VALID_COLORS = frozenset([BLACK_COLOR, EMPTY_COLOR, WHITE_COLOR])


class Rules:
    """
    A class containing rules for a Go game
    """

    @staticmethod
    def row_col_is_valid(row: int, col: int, board_size: int = BOARD_SIZE) -> bool:
        """
        Check if the given row and column is valid

        Args:
            row (int): the row of the move
            col (int): the column of the move
            board_size (int, optional): the size of the game board. Defaults to 9.

        Returns:
            bool: True if the row and column is valid, False otherwise
        """
        if not isinstance(row, int) or not isinstance(col, int):
            return False

        if (row, col) == PASS_MOVE_POSITION:
            return True
        return 0 <= row < board_size and 0 <= col < board_size

    @staticmethod
    def index_is_valid(index: int, board_size: int = BOARD_SIZE) -> bool:
        """
        Check if the given index is valid

        Args:
            index (int): the index of the move
            board_size (int, optional): the size of the game board. Defaults to 9.

        Returns:
            bool: True if the index is valid, False otherwise
        """
        return isinstance(index, int) and 0 <= index < board_size * board_size + 1

    @staticmethod
    def gtp_move_is_valid(gtp_move: str) -> bool:
        """
        Check if the given GTP move is valid

        Args:
            gtp_move (str): the GTP move

        Returns:
            bool: True if the GTP move is valid, False otherwise
        """
        gtp_move = gtp_move.strip().upper()
        if gtp_move == "PASS":
            return True
        if not gtp_move or gtp_move[0] not in "ABCDEFGHJKLMNOPQRSTUVWXYZ":
            return False
        for char in gtp_move[1:]:
            if char not in "1234567890":
                return False
        return True

    @staticmethod
    def color_is_valid(color: int) -> bool:
        """
        Check if the given color is a valid color

        Args:
            color (int): the color of the move

        Returns:
            bool: True if the color is a valid color, False otherwise
        """
        return isinstance(color, int) and color in _VALID_COLORS

    @staticmethod
    def player_name_is_valid(name: str) -> bool:
        """
        Check if the given name is a valid player name

        Args:
            name (str): the name of the player

        Returns:
            bool: True if the name is a valid player name, False otherwise
        """
        return isinstance(name, str)
