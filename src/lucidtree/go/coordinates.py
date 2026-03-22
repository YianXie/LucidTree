from lucidtree.constants import BOARD_SIZE, PASS_INDEX, PASS_MOVE_POSITION
from lucidtree.go.exceptions import InvalidCoordinateError
from lucidtree.go.rules import Rules


def row_col_to_index(row: int, col: int, /) -> int:
    """
    Calculate the index of a move within the encoded tensor

    Args:
        row (int): the row of the move
        col (int): the column of the move

    Returns:
        int: the index within the tensor
    """
    if not Rules.row_col_is_valid(row, col):
        raise InvalidCoordinateError(f"Invalid position: {(row, col)}")

    if (row, col) == PASS_MOVE_POSITION:
        return PASS_INDEX
    return row * BOARD_SIZE + col


def index_to_row_col(index: int, /) -> tuple[int, int]:
    """
    Convert a board index to a move position

    Args:
        index (int): the move index
    """
    if not Rules.index_is_valid(index):
        raise InvalidCoordinateError(f"Invalid index: {index}")

    if index == PASS_INDEX:
        return PASS_MOVE_POSITION

    row, col = divmod(index, BOARD_SIZE)
    return row, col


def gtp_to_row_col(gtp_move: str, /) -> tuple[int, int]:
    """
    Convert a GTP move to a row and column.

    GTP notation skips the letter 'I' to avoid confusion with '1',
    so column letters run A-H, J-T (i.e., 'J' is column 8, not 9).

    Args:
        gtp_move (str): the GTP move

    Returns:
        tuple[int, int]: the row and column

    Raises:
        InvalidCoordinateError: if the GTP move is invalid
    """
    gtp_move = gtp_move.strip().upper()
    if not Rules.gtp_move_is_valid(gtp_move):
        raise InvalidCoordinateError(f"Invalid GTP move: {gtp_move}")

    if gtp_move == "PASS":
        return PASS_MOVE_POSITION

    # Map letter to zero-based column index, skipping 'I'
    letter_index = ord(gtp_move[0]) - ord("A")
    row = int(gtp_move[1:]) - 1
    column = (
        letter_index if letter_index < 8 else letter_index - 1
    )  # skip 'I' (index 8)
    return row, column


def gtp_to_index(gtp_move: str, /) -> int:
    """
    Convert a GTP move to a board index

    Args:
        gtp_move (str): the GTP move

    Returns:
        int: the board index

    Raises:
        InvalidCoordinateError: if the GTP move is invalid
    """
    row, col = gtp_to_row_col(gtp_move)
    return row_col_to_index(row, col)


def row_col_to_gtp(row: int, col: int, /) -> str:
    """
    Convert a row and column to a GTP move.

    GTP notation skips the letter 'I', so column 8 maps to 'J', not 'I'.

    Args:
        row (int): the row
        col (int): the column

    Returns:
        str: the GTP move

    Raises:
        InvalidCoordinateError: if the row or column is invalid
    """
    if not Rules.row_col_is_valid(row, col, BOARD_SIZE):
        raise InvalidCoordinateError(f"Invalid position: {(row, col)}")

    if (row, col) == PASS_MOVE_POSITION:
        return "PASS"

    # Skip 'I': columns 0-7 → A-H, columns 8+ → J-Z
    letter_index = col if col < 8 else col + 1
    letter = chr(ord("A") + letter_index)
    number = str(row + 1)
    return f"{letter}{number}"


def index_to_gtp(index: int, /) -> str:
    """
    Convert a board index to a GTP move

    Args:
        index (int): the board index

    Returns:
        str: the GTP move

    Raises:
        InvalidCoordinateError: if the index is invalid
    """
    if not Rules.index_is_valid(index):
        raise InvalidCoordinateError(f"Invalid index: {index}")

    row, col = index_to_row_col(index)
    return row_col_to_gtp(row, col)
