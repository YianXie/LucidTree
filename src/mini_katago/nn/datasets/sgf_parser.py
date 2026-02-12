import warnings
from pathlib import Path

from sgfmill import sgf

from mini_katago.constants import BLACK_COLOR, WHITE_COLOR
from mini_katago.go.board import Board
from mini_katago.go.game import Game
from mini_katago.go.player import Player


def parse_sgf_file(path: Path) -> Game:
    if not path.exists():
        raise FileNotFoundError("Invalid file path")

    with open(path, "rb") as f:
        sgf_game = sgf.Sgf_game.from_bytes(f.read())

    winner = sgf_game.get_winner()
    if winner is None:
        warnings.warn("Winner attribute not found")

    board_size = sgf_game.get_size()
    root_node = sgf_game.get_root()
    black_player = Player(root_node.get("PB"), BLACK_COLOR)
    white_player = Player(root_node.get("PW"), WHITE_COLOR)
    game_sequence = sgf_game.get_main_sequence()
    board = Board(board_size, black_player, white_player)
    game = Game(
        board,
        black_player,
        white_player,
        black_player if winner == "b" else white_player if winner == "w" else None,
    )

    for idx, node in enumerate(game_sequence):
        color, pos = node.get_move()
        if color is None:
            continue

        if pos is None:
            game.board.pass_move()
        else:
            game.board.place_move(pos, BLACK_COLOR if color == "b" else WHITE_COLOR)

    return game
