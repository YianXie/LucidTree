from mini_katago.constants import BOARD_SIZE, PASS_INDEX, PASS_MOVE_POSITION


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
    Convert a GTP move to a row and column

    Args:
        gtp_move (str): the GTP move
    """
    gtp_move = gtp_move.strip().upper()
    if gtp_move == "PASS":
        return PASS_MOVE_POSITION

    if gtp_move[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        raise ValueError(f"Invalid GTP move: {gtp_move}")

    for char in gtp_move[1:]:
        if char not in "1234567890":
            raise ValueError(f"Invalid GTP move: {gtp_move}")

    # Assuming the letter is the column and the number is the row
    column = ord(gtp_move[0]) - ord("A")
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
    Convert a row and column to a GTP move

    Args:
        row (int): the row
        col (int): the column
    """
    letter = chr(ord("A") + col)
    number = str(row + 1)
    return f"{letter}{number}"


def index_to_gtp(index: int, /) -> str:
    """
    Convert a board index to a GTP move

    Args:
        index (int): the board index
    """
    return row_col_to_gtp(*index_to_row_col(index))
