from pathlib import Path

from sgfmill import sgf

from mini_katago.constants import BLACK_COLOR, WHITE_COLOR
from mini_katago.go.board import Board
from mini_katago.go.player import Player


def parsed_sgf_game_to_board(game: sgf.Sgf_game) -> Board:
    winner = game.get_winner()
    board_size = game.get_size()
    root_node = game.get_root()
    black_player = Player(root_node.get("PB"), BLACK_COLOR)
    white_player = Player(root_node.get("PW"), WHITE_COLOR)
    game_sequence = game.get_main_sequence()
    board = Board(board_size, black_player, white_player)

    if winner is not None:
        board.set_winner(black_player if winner == "b" else white_player)

    for node in game_sequence:
        color, pos = node.get_move()
        if color is None:
            continue
        if pos is None:
            board.pass_move()
        else:
            board.place_move(pos, BLACK_COLOR if color == "b" else WHITE_COLOR)

    return board


def parse_sgf_file(path: Path) -> Board:
    if not path.exists():
        raise FileNotFoundError("Invalid file path")

    with open(path, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    return parsed_sgf_game_to_board(game)


# For testing
if __name__ == "__main__":
    path = Path("./src/mini_katago/data/")
    for sgf_file in path.iterdir():
        board = parse_sgf_file(
            sgf_file
        )  # Do note that some sgf file may cause illegal move error
        print(board)
