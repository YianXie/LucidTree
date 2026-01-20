from mini_katago.constants import EMPTY_COLOR

from .rules import Rules


class Move:
    """
    A class representing a move
    """

    def __init__(
        self,
        row: int = -1,
        col: int = -1,
        color: int = EMPTY_COLOR,
        *,
        passed: bool = False,
    ) -> None:
        """
        Initialize the move

        Args:
            row (int, optional): the row of the move. Defaults to -1.
            col (int, optional): the column of the move. Defaults to -1.
            color (int, optional): the color of the move. Defaults to 0.
            passed (bool, optional): whether the move is passed. Defaults to False.
        """
        self.row = row
        self.col = col
        self.color = color
        self.passed = passed

    def set_color(self, color: int) -> None:
        """
        Set the color of the move

        Args:
            color (int): the color of the move

        Raises:
            ValueError: if the color is invalid
        """
        if not Rules.color_is_valid(color):
            raise ValueError(f"Invalid color: {color}")
        self.color = color

    def get_position(self) -> tuple[int, int]:
        """
        Get the position of the move

        Returns:
            tuple: the position of the move
        """
        return (self.row, self.col)

    def get_color(self) -> int:
        """
        Get the color of the move

        Returns:
            int: the color of the move
        """
        return self.color

    def is_empty(self) -> bool:
        """
        Check if a move is still empty

        Returns:
            bool: True if it is empty, False otherwise
        """
        return self.color == 0

    def is_passed(self) -> bool:
        """
        Check if the move is passed

        Returns:
            bool: True if the move is passed, False otherwise
        """
        return self.passed

    def __hash__(self) -> int:
        """
        Return the hash of the move based on its position and color

        Returns:
            int: the hash value
        """
        return hash((self.row, self.col, self.passed))

    def __eq__(self, other: object) -> bool:
        """
        Check if two moves are equal

        Args:
            other: the other object to compare with

        Returns:
            bool: True if the moves are equal, False otherwise
        """
        if not isinstance(other, Move):
            return NotImplemented
        return (self.row, self.col, self.passed) == (other.row, other.col, other.passed)

    def __lt__(self, other: object) -> bool:
        """
        Compare two moves for ordering

        Args:
            other: the other Move to compare with

        Returns:
            bool: True if this move is less than the other, False otherwise

        Raises:
            TypeError: if other is not a Move
        """
        if not isinstance(other, Move):
            return NotImplemented

        # Compare by row first, then col, then passed
        if self.row != other.row:
            return self.row < other.row
        if self.col != other.col:
            return self.col < other.col
        return self.passed < other.passed

    def __repr__(self) -> str:
        """
        Return a developer-friendly message

        Returns:
            str: a message that describes the move
        """
        return f"(({self.row}, {self.col}), {self.color})"
