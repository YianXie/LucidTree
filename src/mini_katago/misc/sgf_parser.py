from pathlib import Path
from typing import Any

from sgfmill import sgf

from mini_katago.go.board import Board


def parsed_nodes_to_board(nodes) -> Board:
    raise NotImplementedError


def parse_sgf_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("Invalid file path")

    with open(path, "rb", encoding="utf-8") as f:
        game = sgf.Sgf_game.from_bytes(f)
    winner = game.get_winner()
    board_size = game.get_size()
    root_node = game.get_root()
    black_player = root_node.get("PB")
    white_player = root_node.get("PW")
    game_sequence = game.get_main_sequence()

    return {
        "winner": winner,
        "board_size": board_size,
        "root_node": root_node,
        "black_player": black_player,
        "white_player": white_player,
        "game_sequence": game_sequence,
    }
