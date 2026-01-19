import math

BOARD_SIZE = 9
"""
The default board size
"""

EMPTY_COLOR = 0
"""
The color of the empty space
"""

BLACK_COLOR = 1
"""
The color of the black player
"""

WHITE_COLOR = -1
"""
The color of the white player
"""

KOMI = 7.5
"""
The Komi value for the game
"""

INFINITY = math.inf
"""
Represents a very large value
"""

EXPLORATION_CONSTANT = 1.5
"""
The 'C' value in MCTS
"""

NUM_SIMULATIONS = 1000
"""
The amount of simulations to do for MCTS
"""

MAX_GAME_DEPTH = 50
"""
The maximum depth of the game
"""

CAPTURE_BOOST = 6.0
"""
The boost for captures
"""

ADJ_BOOST = 1.8
"""
The boost for adjacent stones
"""

CHANNEL_SIZE = 6
"""
The channel size for the encoded board in encode_position
"""

PASS_INDEX = 81
"""
The index of a pass for the CNN Tensor
"""
