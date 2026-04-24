# fmt: off

from typing import Any

import torch

from api.common.utils import build_board_from_request, parse_player
from lucidtree.constants import BLACK_COLOR, KOMI, RULES
from lucidtree.engine.analysis import analyze_position
from lucidtree.engine.winrate import generate_winrate
from lucidtree.go.player import Player
from lucidtree.nn.agent import load_model

# fmt: on


def analyze(validated_data: dict[str, Any], /) -> dict[str, Any]:
    """
    Analyze a position from a request

    Args:
        validated_data (dict[str, Any]): the validated data

    Returns:
        dict[str, Any]: the analyzed position
    """
    algo = validated_data["algo"]
    komi = validated_data.get("komi", KOMI)
    moves = validated_data.get("moves", [])
    rules = validated_data.get("rules", RULES)
    to_play_text = validated_data["to_play"]
    params = validated_data["params"]
    output = validated_data["output"]

    board = build_board_from_request(moves=moves)
    to_play = parse_player(to_play_text)
    opponent = Player.white() if to_play.get_color() == BLACK_COLOR else Player.black()
    to_play.opponent = opponent
    opponent.opponent = to_play

    return analyze_position(
        algo=algo,
        board=board,
        komi=komi,
        rules=rules,
        to_play=to_play,
        params=params,
        output=output,
    )


def winrate(validated_data: dict[str, Any], /) -> dict[str, Any]:
    """
    Generate the winrate data for a given game

    Args:
        validated_data (dict[str, Any]): the validated data

    Returns:
        dict[str, Any]: the winrate
    """
    moves = validated_data.get("moves", [])
    params = validated_data.get("params", {})

    device = params.get("device", None)
    temperature = params.get("temperature", 0.0)

    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    model = load_model(device=device)

    return {
        "winrate": generate_winrate(
            moves, device=device, temperature=temperature, model=model
        )
    }
