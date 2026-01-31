from .rules import Rules


class Player:
    """
    A class representing a player
    """

    def __init__(self, name: str, color: int) -> None:
        """
        Initialize a player

        Args:
            name (str): the player's name
            color (int): the player's color should be either -1 (black) or 1 (white)
        """
        self.name = name
        self.color = color
        self.capture_count = 0
        self.opponent: Player | None = None

    def set_name(self, name: str) -> None:
        """
        Set the name of the player

        Args:
            name (str): the new name
        """
        if not Rules.player_name_is_valid(name):
            raise TypeError("Invalid name type: expecting a string")
        self.name = name

    def set_color(self, color: int) -> None:
        """
        Set the color of the player

        Args:
            color (int): the color of the player, should be either -1 (black) or 1 (white)

        Raises:
            ValueError: if the color is not valid
        """
        if not Rules.color_is_valid(color):
            raise ValueError(f"Invalid color: {color}")
        self.color = color

    def set_capture_count(self, amount: int) -> None:
        """
        Set the amount of captures for a player

        Args:
            amount (int): the new amount of captures
        """
        self.capture_count = amount

    def increase_capture_count(self, amount: int) -> None:
        """
        Add the amount of captures for a player

        Args:
            amount (int): the amount to add
        """
        self.capture_count += amount

    def get_name(self) -> str:
        """
        Return the name of the player

        Returns:
            str: the player's name
        """
        return self.name

    def get_color(self) -> int:
        """
        Return the color of the player

        Returns:
            int: the player's color
        """
        return self.color

    def get_capture_count(self) -> int:
        """
        Return the capture count of the player

        Returns:
            int: the capture count
        """
        return self.capture_count

    def __repr__(self) -> str:
        return f"player name: {self.name}, player color: {self.color}"
