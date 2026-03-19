from typing import Any

from common.exceptions import BadRequestError

from mini_katago.constants import BLACK_COLOR, PASS_MOVE_POSITION
from mini_katago.engine.analysis import analyze_position
from mini_katago.go.board import Board
from mini_katago.go.coordinates import gtp_to_row_col
from mini_katago.go.player import Player


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
    Parse a move from a string

    Args:
        value (str): the move string, should be in the format 'A1', 'B2', etc. or 'PASS'

    Raises:
        BadRequestError: if the move is not valid, expecting 'A1', 'B2', etc. or 'PASS'

    Returns:
        tuple[int, int]: the move position
    """
    try:
        return gtp_to_row_col(value)
    except Exception as e:
        raise BadRequestError(f"Invalid move: {e}")


def _build_board_from_request(
    *, board_size: int, moves: list[tuple[str, str]]
) -> Board:
    """
    Build a board from a request moves

    Args:
        board_size (int): the board size
        moves (list[tuple[str, str]]): the moves

    Returns:
        Board: the board object
    """
    board = Board(board_size, Player.black(), Player.white())

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
    board_size = validated_data["board_size"]
    to_play_text = validated_data["to_play"]
    moves = validated_data.get("moves", [])
    algo = validated_data["algo"]
    params = validated_data.get("params", {})

    board = _build_board_from_request(board_size=board_size, moves=moves)
    to_play = _parse_player(to_play_text)
    if to_play.get_color() == BLACK_COLOR:
        to_play.opponent = Player.white()
    else:
        to_play.opponent = Player.black()

    return analyze_position(board=board, to_play=to_play, algo=algo, params=params)
