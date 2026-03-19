import random

from lucidtree.go.game import Game


def split_game(
    games: list[Game],
    *,
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

    Raises:
        ValueError: if the split ratios do not sum to 1.0
    """
    if abs(train + val + test - 1.0) >= 1e-6:
        raise ValueError(
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
