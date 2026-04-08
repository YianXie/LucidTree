# fmt: off

import time

import torch

from lucidtree.constants import (BLACK_COLOR, BOARD_SIZE, PASS_MOVE_POSITION,
                                 WHITE_COLOR)
from lucidtree.go.interactive_board import InteractiveBoard
from lucidtree.go.player import Player
from lucidtree.nn.agent import load_model, pick_moves_mcts, pick_moves_nn

# fmt: on


def human_vs_nn() -> None:
    """
    Initialize a human v.s. nn game
    """
    board = InteractiveBoard(
        BOARD_SIZE,
        Player("Black player", BLACK_COLOR),
        Player("White player", WHITE_COLOR),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device)
    board.start_display()

    while not board.is_terminate():
        # Process pygame events (handles user input)
        board.process_events()

        # Check if user closed the window
        if not board._is_displayed:
            break

        # Small delay to prevent excessive CPU usage
        time.sleep(0.1)

        # AI's turn (White)
        if board.current_player.get_color() == WHITE_COLOR:
            best, prob, value = pick_moves_nn(model=model, board=board, device=device)
            print(f"Move probability: {prob}")
            print(f"Value: {value}")

            if best == PASS_MOVE_POSITION:
                board.pass_move()
            else:
                board.place_move(best, WHITE_COLOR)

    board.stop_display()


def human_vs_mcts_nn() -> None:
    black_player = Player("Black player", BLACK_COLOR)
    white_player = Player("White player", WHITE_COLOR)
    black_player.opponent, white_player.opponent = white_player, black_player
    board = InteractiveBoard(
        BOARD_SIZE,
        black_player,
        white_player,
    )
    board.start_display()

    while not board.is_terminate():
        # Process pygame events (handles user input)
        board.process_events()

        # Check if user closed the window
        if not board._is_displayed:
            break

        # Small delay to prevent excessive CPU usage
        time.sleep(0.1)

        # AI's turn (White)
        if board.current_player.get_color() == WHITE_COLOR:
            # Pass a plain Board copy - MCTS deepcopies boards and InteractiveBoard
            # contains pygame.Surface objects that cannot be deepcopied
            pos = pick_moves_mcts(board=board.copy_game_state(), to_play=white_player)
            if pos == PASS_MOVE_POSITION:
                print("AI passed")
                board.pass_move()
            else:
                board.place_move(pos, WHITE_COLOR)

    board.stop_display()


def main() -> None:
    human_vs_mcts_nn()
