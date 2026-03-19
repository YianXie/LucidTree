"""
A file for testing
"""

from lucidtree.constants import BLACK_COLOR, BOARD_SIZE, WHITE_COLOR
from lucidtree.go.interactive_board import InteractiveBoard
from lucidtree.go.player import Player

black_player = Player("Black Player", BLACK_COLOR)
white_player = Player("White Player", WHITE_COLOR)
board = InteractiveBoard(BOARD_SIZE, black_player, white_player)
board.show_board()
