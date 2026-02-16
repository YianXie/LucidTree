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
    board.show_interactive_board()
except Exception as e:
    print("Error:", e)
