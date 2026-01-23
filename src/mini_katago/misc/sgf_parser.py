from pathlib import Path

from sgfmill import sgf

from mini_katago.constants import BLACK_COLOR, WHITE_COLOR
from mini_katago.go.board import Board
from mini_katago.go.game import Game
from mini_katago.go.player import Player


def parsed_sgf_game_to_game(sgf_game: sgf.Sgf_game) -> Game:
    winner = sgf_game.get_winner()
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
        None if winner is None else black_player if winner == "b" else white_player,
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
    if not path.exists():
        raise FileNotFoundError("Invalid file path")

    with open(path, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    return parsed_sgf_game_to_game(game)


# For testing
if __name__ == "__main__":
    path = Path("./src/mini_katago/data/")
    for sgf_file in path.iterdir():
        try:
            game = parse_sgf_file(sgf_file)
            print(game)
        except Exception as e:
            print("Error:", e)
