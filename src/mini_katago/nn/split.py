import random

from mini_katago.go.game import Game
from mini_katago.utils import get_project_root


def split_game(
    games: list[Game],
    seed: int | None = 0,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
) -> tuple[list[Game], list[Game], list[Game]]:
    """
    Split the games to train, val, and test group

    Args:
        games (list[Game]): the games
        seed (int | None, optional): the random seed. Defaults to 0.
        train (float, optional): the percentage of games for training. Defaults to 0.8.
        val (float, optional): the percentage of games for validation. Defaults to 0.1.
        test (float, optional): the percentage of games for testing. Defaults to 0.1.

    Returns:
        tuple[list[Game], list[Game], list[Game]]: the splitted games
    """
    assert abs(train + val + test - 1.0) < 1e-6, (
        f"Split ratios must sum to 1.0 (within 1e-6), got {train + val + test}"
    )

    rng = random.Random(seed)
    games = games[:]
    rng.shuffle(games)

    n = len(games)
    n_train = int(n * train)
    n_val = int(n * val)

    train_games = games[:n_train]
    val_games = games[n_train : n_train + n_val]
    test_games = games[n_train + n_val :]

    return train_games, val_games, test_games


def filter_games() -> None:
    for f in (get_project_root() / "data/raw/sgf").glob("*.sgf"):
        text = f.read_text(errors="ignore")
        if "RE[" not in text:
            f.unlink()


def rename_games() -> None:
    cnt = 1
    for f in (get_project_root() / "data/raw/sgf").glob("*.sgf"):
        parent = f.parent
        new_f = parent / f"{cnt:0{5}d}.sgf"
        f.rename(new_f)
        cnt += 1


if __name__ == "__main__":
    rename_games()
