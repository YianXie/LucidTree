from typing import Any

from api.api import settings
from api.common.exceptions import BadRequestError
from lucidtree.constants import BLACK_COLOR, BOARD_SIZE, PASS_MOVE_POSITION
from lucidtree.engine.analysis import analyze_position
from lucidtree.go.board import Board
from lucidtree.go.coordinates import gtp_to_row_col
from lucidtree.go.player import Player


def _parse_player(value: str, /) -> Player:
    """
    Parse a player from a string

    Args:
        value (str): the player string, should be either 'B' or 'W'

    Raises:
        BadRequestError: if the player is not valid, expecting 'B' or 'W'

    Returns:
        Player: the player object
    """
    value = value.upper()
    if value == "B":
        return Player.black()
    elif value == "W":
        return Player.white()
    else:
        raise BadRequestError(f"Invalid player: expecting 'B' or 'W', got '{value}'")


def _parse_move(value: str, /) -> tuple[int, int]:
    """
    Parse a move from a string and validate it lies within the board.

    Args:
        value (str): the move string, should be in the format 'A1', 'B2', etc. or 'PASS'

    Raises:
        BadRequestError: if the move is not valid, expecting 'A1', 'B2', etc. or 'PASS'

    Returns:
        tuple[int, int]: the move position
    """
    try:
        position = gtp_to_row_col(value)
    except Exception as e:
        raise BadRequestError(f"Invalid move: {e}")

    if position != PASS_MOVE_POSITION:
        row, col = position
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            raise BadRequestError(
                f"Move '{value}' is out of bounds for a {BOARD_SIZE}x{BOARD_SIZE} board"
            )

    return position


def _build_board_from_request(moves: list[tuple[str, str]]) -> Board:
    """
    Build a board from a request moves

    Args:
        moves (list[tuple[str, str]]): the moves

    Returns:
        Board: the board object
    """
    board = Board(BOARD_SIZE, Player.black(), Player.white())

    for color_text, point_text in moves:
        player = _parse_player(color_text)
        move_position = _parse_move(point_text)

        try:
            if move_position == PASS_MOVE_POSITION:
                board.pass_move()
            else:
                board.place_move(move_position, player.get_color())
        except Exception as e:
            raise BadRequestError(f"Invalid move: {e}")

    return board


def analyze(validated_data: dict[str, Any], /) -> dict[str, Any]:
    """
    Analyze a position from a request

    Args:
        validated_data (dict[str, Any]): the validated data

    Returns:
        dict[str, Any]: the analyzed position
    """
    to_play_text = validated_data["to_play"]
    moves = validated_data.get("moves", [])
    algo = validated_data["algo"]
    params = validated_data.get("params", {})

    board = _build_board_from_request(moves=moves)
    to_play = _parse_player(to_play_text)
    opponent = Player.white() if to_play.get_color() == BLACK_COLOR else Player.black()
    to_play.opponent = opponent
    opponent.opponent = to_play

    return analyze_position(
        board=board,
        to_play=to_play,
        algo=algo,
        params=params,
        model=settings.MODEL_PATH,
    )
