class InvalidMoveError(Exception):
    """
    Exception raised for invalid moves
    """

    pass


class IllegalMoveError(Exception):
    """
    Exception raised for illegal moves
    """

    pass


class GameOverError(Exception):
    """
    Exception raised for game over
    """

    pass


class InvalidCoordinateError(Exception):
    """
    Exception raised for invalid coordinates
    """

    pass


class InvalidColorError(Exception):
    """
    Exception raised for invalid colors
    """

    pass


class InvalidNameError(Exception):
    """
    Exception raised for invalid names
    """

    pass


class BadRequestError(Exception):
    """
    Exception raised for bad requests
    """

    pass
