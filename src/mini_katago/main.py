"""
A file for testing
"""

from mini_katago.constants import BLACK_COLOR, BOARD_SIZE, WHITE_COLOR
from mini_katago.go.board import Board
from mini_katago.go.player import Player

black_player = Player("Black Player", BLACK_COLOR)
white_player = Player("White Player", WHITE_COLOR)
board = Board(BOARD_SIZE, black_player, white_player)
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
