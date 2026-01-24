import random

from mini_katago.go.game import Game


def split_game(
    games: list[Game],
    seed: int = 0,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
) -> tuple[list[Game], list[Game], list[Game]]:
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
