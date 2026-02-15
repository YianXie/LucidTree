import warnings
from pathlib import Path

from sgfmill import sgf

from mini_katago.constants import BLACK_COLOR, WHITE_COLOR
from mini_katago.go.board import Board
from mini_katago.go.game import Game
from mini_katago.go.player import Player


def parse_sgf_game(sgf_game: sgf.Sgf_game) -> Game:
    """
    Parse an in-memory SGF game object into a Game.

    Args:
        sgf_game: An sgfmill Sgf_game object (e.g. from Sgf_game.from_bytes or Sgf_game(size=...))

    Returns:
        Game: The parsed game with board, players, and winner (if specified).
    """
    winner = sgf_game.get_winner()
    if winner is None:
        warnings.warn("Winner attribute not found")

    board_size = sgf_game.get_size()
    black_player = Player("Black", BLACK_COLOR)
    white_player = Player("White", WHITE_COLOR)
    game_sequence = sgf_game.get_main_sequence()
    board = Board(board_size, black_player, white_player)
    game = Game(
        board,
        black_player,
        white_player,
        black_player if winner == "b" else white_player if winner == "w" else None,
    )

    for node in game_sequence:
        color, pos = node.get_move()
        if color is None:
            continue

        if pos is None:
            game.board.pass_move()
        else:
            game.board.place_move(pos, BLACK_COLOR if color == "b" else WHITE_COLOR)

    return game


def parse_sgf_file(path: Path) -> Game:
    """
    Parse a given sgf file into game

    Args:
        path (Path): the path to the game file

    Raises:
        FileNotFoundError: if the file path is invalid

    Returns:
        Game: the parsed game
    """
    if not path.exists():
        raise FileNotFoundError("Invalid file path")

    with open(path, "rb") as f:
        sgf_game = sgf.Sgf_game.from_bytes(f.read())

    return parse_sgf_game(sgf_game)
