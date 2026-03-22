from lucidtree.constants import BOARD_SIZE, PASS_INDEX, PASS_MOVE_POSITION


def move_to_index(move_position: tuple[int, int] | None, /) -> int:
    """
    Calculate the index of a move within the encoded tensor

    Args:
        move_position (tuple[int, int] | None): the move's position

    Returns:
        int: the index within the tensor
    """
    if move_position == PASS_MOVE_POSITION or move_position is None:
        return PASS_INDEX

    row, col = move_position
    return row * BOARD_SIZE + col


def index_to_row_col(index: int, /) -> tuple[int, int]:
    """
    Convert a board index to a move position

    Args:
        index (int): the move index
    """
    if index == PASS_INDEX:
        return (-1, -1)

    return divmod(index, BOARD_SIZE)


def gtp_to_row_col(gtp_move: str, /) -> tuple[int, int]:
    """
    Convert a GTP move to a row and column.

    GTP notation skips the letter 'I' to avoid confusion with '1',
    so column letters run A-H, J-T (i.e., 'J' is column 8, not 9).

    Args:
        gtp_move (str): the GTP move
    """
    gtp_move = gtp_move.strip().upper()
    if gtp_move == "PASS":
        return PASS_MOVE_POSITION

    # GTP skips 'I'; valid column letters are A-H and J-Z
    valid_columns = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    if not gtp_move or gtp_move[0] not in valid_columns:
        raise ValueError(f"Invalid GTP move: {gtp_move}")

    for char in gtp_move[1:]:
        if char not in "1234567890":
            raise ValueError(f"Invalid GTP move: {gtp_move}")

    # Map letter to zero-based column index, skipping 'I'
    letter = gtp_move[0]
    letter_index = ord(letter) - ord("A")
    column = letter_index if letter_index < 8 else letter_index - 1  # skip 'I' (index 8)
    row = int(gtp_move[1:]) - 1
    return row, column


def gtp_to_index(gtp_move: str, /) -> int:
    """
    Convert a GTP move to a board index

    Args:
        gtp_move (str): the GTP move
    """
    return move_to_index(gtp_to_row_col(gtp_move))


def row_col_to_gtp(row: int, col: int, /) -> str:
    """
    Convert a row and column to a GTP move.

    GTP notation skips the letter 'I', so column 8 maps to 'J', not 'I'.

    Args:
        row (int): the row
        col (int): the column
    """
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
    """
    return row_col_to_gtp(*index_to_row_col(index))
