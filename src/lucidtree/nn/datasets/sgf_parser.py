import logging
import warnings
from pathlib import Path
from typing import Any

from sgfmill import sgf

from lucidtree.constants import BLACK_COLOR, WHITE_COLOR
from lucidtree.go.board import Board
from lucidtree.go.game import Game
from lucidtree.go.player import Player


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
    black_player_name = sgf_game.get_player_name("b")
    white_player_name = sgf_game.get_player_name("w")
    black_player = Player(
        black_player_name if black_player_name is not None else "Black", BLACK_COLOR
    )
    white_player = Player(
        white_player_name if white_player_name is not None else "White", WHITE_COLOR
    )
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
        RuntimeError: if the file path does not point to a file
        RuntimeError: if the file has an incorrect extension name

    Returns:
        Game: the parsed game
    """
    if not path.exists():
        raise FileNotFoundError("Invalid file path")
    if not path.is_file():
        raise RuntimeError("Invalid path. Expects a file.")
    if not path.suffix == ".sgf":
        raise RuntimeError("Incorrect extension name. Expects a .sgf file.")

    with open(path, "rb") as f:
        sgf_game = sgf.Sgf_game.from_bytes(f.read())

    return parse_sgf_game(sgf_game)


def parse_sgf_files(
    path: Path,
    *,
    graceful: bool = True,
    start: int = 0,
    amount: int | None = None,
    **kwargs: Any,
) -> list[Game]:
    """
    Parse sgf files from a given directory

    Args:
        path (Path): the path to the directory
        graceful (bool, optional): if the function should handle exceptions gracefully. Defaults to True.
        start (int, optional): the starting sgf file. Defaults to 0.
        amount (int | None, optional): the amount ot parse. Defaults to None.

    Raises:
        FileNotFoundError: if the directory is not found
        NotADirectoryError: if the given path is not a directory
        RuntimeError: if the logger is not given but log is True
        e: exceptions when parsing games

    Returns:
        list[Game]: the parsed games
    """
    if not path.exists():
        raise FileNotFoundError("Target directory not found.")
    if not path.is_dir():
        raise NotADirectoryError("Expects a directory, not a file.")

    log: bool = kwargs.get("log", False)
    logger: logging.Logger | None = kwargs.get("logger", None)
    gap: int = kwargs.get("gap", 1000)

    if logger is None and log is True:
        raise RuntimeError("Logger must be specified if log is True.")

    sgf_files = path.glob("*.sgf")
    games: list[Game] = []

    game_parsed = 0
    for idx, sgf_file in enumerate(sgf_files):
        if idx < start:
            continue

        try:
            games.append(parse_sgf_file(sgf_file))
            game_parsed += 1
        except Exception as e:
            if log:
                logger.warning("Skipped file: %s. ValueError: %s.", sgf_file, e)  # type: ignore
            if not graceful:
                raise e

        if log and (idx + 1) % gap == 0:
            logger.info(  # type: ignore
                "Attempted to parse %d games. %d games parsed successfully.",
                game_parsed,
            )
        if amount is not None and idx + 1 >= start + amount:
            break

    return games
