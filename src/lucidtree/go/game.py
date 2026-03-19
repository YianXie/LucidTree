from dataclasses import dataclass

from lucidtree.go.board import Board
from lucidtree.go.player import Player


@dataclass
class Game:
    """
    A data class that represents a Go game
    """

    board: Board
    black_player: Player
    white_player: Player
    winner: Player | None = None
