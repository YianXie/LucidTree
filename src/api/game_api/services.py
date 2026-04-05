from typing import Any

from api.api import settings
from api.common.exceptions import BadRequestError
from lucidtree.constants import BLACK_COLOR, BOARD_SIZE, PASS_MOVE_POSITION
from lucidtree.engine.analysis import analyze_position
from lucidtree.go.board import Board
from lucidtree.go.coordinates import gtp_to_row_col
from lucidtree.go.player import Player


def _merge_analysis_config_into_params(
    algo: str, params: dict[str, Any], analysis_config: dict[str, Any]
) -> dict[str, Any]:
    """Flatten LucidGo-style nested analysis_config into LucidTree runtime params."""
    merged = dict(params)

    general = (
        analysis_config.get("general", {}) if isinstance(analysis_config, dict) else {}
    )
    neural_network = (
        analysis_config.get("neural_network", {})
        if isinstance(analysis_config, dict)
        else {}
    )
    mcts = analysis_config.get("mcts", {}) if isinstance(analysis_config, dict) else {}
    minimax = (
        analysis_config.get("minimax", {}) if isinstance(analysis_config, dict) else {}
    )
    output = (
        analysis_config.get("output", {}) if isinstance(analysis_config, dict) else {}
    )

    if "seed" in general:
        merged.setdefault("seed", general["seed"])
    if "temperature" in general:
        merged.setdefault("temperature", general["temperature"])
    if "max_time_ms" in general:
        merged.setdefault("max_time_ms", general["max_time_ms"])

    if algo == "nn":
        if "model" in neural_network:
            merged.setdefault("model", neural_network["model"])
        if "policy_softmax_temperature" in neural_network:
            merged.setdefault(
                "policy_softmax_temperature",
                neural_network["policy_softmax_temperature"],
            )
        if "use_value_head" in neural_network:
            merged.setdefault("use_value_head", neural_network["use_value_head"])
    elif algo == "mcts":
        for key in (
            "num_simulations",
            "c_puct",
            "dirichlet_alpha",
            "dirichlet_epsilon",
            "value_weight",
            "policy_weight",
            "select_by",
        ):
            if key in mcts:
                merged.setdefault(key, mcts[key])
        if "model" in neural_network:
            merged.setdefault("model", neural_network["model"])
    elif algo == "minimax":
        for key in ("depth", "use_alpha_beta"):
            if key in minimax:
                merged.setdefault(key, minimax[key])

    for key in ("include_top_moves", "include_policy", "include_win_rate"):
        if key in output:
            merged.setdefault(key, output[key])

    return merged


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
    analysis_config = validated_data.get("analysis_config", {})

    params = _merge_analysis_config_into_params(
        algo=algo,
        params=params,
        analysis_config=analysis_config,
    )

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
