from pathlib import Path

from sgfmill import sgf


def filter_games(path: Path) -> None:
    """
    Filter out valid SGF games, and delete invalid ones

    Args:
        path (Path): the path to the SGF directory

    Raises:
        FileNotFoundError: if the directory does not exist
        RuntimeError: if the given path is not a directory
    """
    if not path.exists():
        raise FileNotFoundError("Given path does not exist.")
    if not path.is_dir():
        raise RuntimeError("Given path must be a directory")

    for file in path.glob("*.sgf"):
        with open(file, "rb") as f:
            try:
                sgf_game = sgf.Sgf_game.from_bytes(f.read())
            except ValueError:
                file.unlink()
                continue

        winner = sgf_game.get_winner()
        if winner is None:
            file.unlink()
            continue


def rename_games(path: Path) -> None:
    """
    Rename all SGF games in a directory in numerical order

    Args:
        path (Path): the path to the directory containing all the SGF games

    Raises:
        FileNotFoundError: if the directory does not exist
        RuntimeError: if the given path is not a directory
    """
    if not path.exists():
        raise FileNotFoundError("Given path does not exist.")
    if not path.is_dir():
        raise RuntimeError("Given path must be a directory")

    cnt = 0
    for file in path.glob("*.sgf"):
        parent = file.parent
        new_f = parent / f"{cnt:0{5}d}.sgf"
        file.rename(new_f)
        cnt += 1
