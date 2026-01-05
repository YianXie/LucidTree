"""
A file for testing
"""

from sgfmill import sgf

from mini_katago.board import Board
from mini_katago.constants import BLACK_COLOR, WHITE_COLOR
from mini_katago.player import Player

with open("src/mini_katago/data/test_territories.sgf", "rb") as f:
    game = sgf.Sgf_game.from_bytes(f.read())
root_node = game.get_root()
black_player = Player(root_node.get("PB"), BLACK_COLOR)
white_player = Player(root_node.get("PW"), WHITE_COLOR)
# moves = [node.get_move() for node in game.get_main_sequence()]

board = Board(9, black_player, white_player)
color = BLACK_COLOR

try:
    while not board.is_terminate():
        row, col = map(int, input("Enter the position to play: ").split())
        if row == -1 or col == -1:
            board.pass_move()
        elif row == -2 or col == -2:
            board.undo()
        else:
            board.place_move((row, col), color)
        color *= -1
        board.print_ascii_board()
except Exception as e:
    print("Error:", e)

# for move in moves:
#     if None in move:
#         continue
#     color, row, col = move[0], move[1][0], move[1][1]
#     board.place_move((row, col), -1 if color == "b" else 1)

# print(board.calculate_score())
# board.show_board()
