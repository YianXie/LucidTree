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

    return divmod(index, 19)


def gtp_to_row_col(gtp_move: str, /) -> tuple[int, int]:
    """
    Convert a GTP move to a row and column

    Args:
        gtp_move (str): the GTP move
    """
    if gtp_move == "pass":
        return (-1, -1)

    # Assuming the letter is the column and the number is the row
    column = ord(gtp_move[0]) - ord("a")
    row = int(gtp_move[1:]) - 1
    return row, column


def gtp_to_index(gtp_move: str, /) -> int:
    """
    Convert a GTP move to a board index

    Args:
        gtp_move (str): the GTP move
    """
    return move_to_index(gtp_to_row_col(gtp_move))
