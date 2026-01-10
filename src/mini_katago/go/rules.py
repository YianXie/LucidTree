from mini_katago.constants import BLACK_COLOR, EMPTY_COLOR, WHITE_COLOR

# Pre-compute valid colors set for faster lookup
_VALID_COLORS = frozenset([BLACK_COLOR, EMPTY_COLOR, WHITE_COLOR])


class Rules:
    """
    A class containing rules for a Go game
    """

    @staticmethod
    def position_is_valid(position: tuple[int, int], size: int = 9) -> bool:
        """
        Check if the given position is valid

        Args:
            position (tuple): the position of the move
            size (int, optional): the size of the game board. Defaults to 9.

        Raises:
            TypeError: if the position argument is not a tuple

        Returns:
            bool: True if the move is valid, False otherwise
        """
        if not isinstance(position, tuple):
            return False
        return 0 <= position[0] < size and 0 <= position[1] < size

    @staticmethod
    def color_is_valid(color: int) -> bool:
        """
        Check if the given color is a valid color

        Args:
            color (int): the color of the move

        Raises:
            TypeError: if the color argument is not an int

        Returns:
            bool: True if the color is a valid color, False otherwise
        """
        if not isinstance(color, int):
            return False
        return color in _VALID_COLORS

    @staticmethod
    def player_name_is_valid(name: str) -> bool:
        if not isinstance(name, str):
            return False
        return True
