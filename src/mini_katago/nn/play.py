# fmt: off

import time

import torch

from mini_katago.constants import (BLACK_COLOR, BOARD_SIZE, PASS_MOVE_POSITION,
                                   WHITE_COLOR)
from mini_katago.go.board import Board
from mini_katago.go.player import Player
from mini_katago.nn.agent import load_model, pick_move, pick_move_mcts

# fmt: on


def human_vs_nn() -> None:
    """
    Initialize a human v.s. nn game
    """
    board = Board(
        BOARD_SIZE,
        Player("Black player", BLACK_COLOR),
        Player("White player", WHITE_COLOR),
    )

    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while not board.is_terminate():
        row, col = map(int, input("Enter your move: ").split())
        if row == -1 and col == -1:
            board.show_board()
            break
        board.place_move((row, col), BLACK_COLOR)
        board.print_ascii_board()

        best, prob, value = pick_move(model=model, board=board, device=device)
        print(f"Move probability: {prob}")
        print(f"Value: {value}")

        if best is not None:
            board.place_move(best, WHITE_COLOR)
        else:
            board.pass_move()
        board.print_ascii_board()


def human_vs_mcts_nn() -> None:
    black_player = Player("Black player", BLACK_COLOR)
    white_player = Player("White player", WHITE_COLOR)
    black_player.opponent, white_player.opponent = white_player, black_player
    board = Board(
        BOARD_SIZE,
        black_player,
        white_player,
    )

    while not board.is_terminate():
        row, col = map(int, input("Enter your move: ").split())
        if row == -1 and col == -1:
            board.show_board()
            break
        board.place_move((row, col), BLACK_COLOR)
        board.print_ascii_board()

        pos = pick_move_mcts(board=board, to_play=white_player)
        if pos == PASS_MOVE_POSITION:
            print("AI passed")
            board.pass_move()
        else:
            board.place_move(pos, WHITE_COLOR)
        board.print_ascii_board()


def mcts_nn_vs_mcts_nn() -> None:
    black_player = Player("Black player", BLACK_COLOR)
    white_player = Player("White player", WHITE_COLOR)
    black_player.opponent, white_player.opponent = white_player, black_player
    board = Board(
        BOARD_SIZE,
        black_player,
        white_player,
    )

    while not board.is_terminate():
        pos = pick_move_mcts(board=board, to_play=white_player)
        if pos == PASS_MOVE_POSITION:
            print("Black passed")
            board.pass_move()
        else:
            board.place_move(pos, BLACK_COLOR)
        board.print_ascii_board()

        pos = pick_move_mcts(board=board, to_play=white_player)
        if pos == PASS_MOVE_POSITION:
            print("White passed")
            board.pass_move()
        else:
            board.place_move(pos, WHITE_COLOR)
        board.print_ascii_board()


def nn_vs_nn() -> None:
    """
    Initialize a nn v.s. nn game
    """
    board = Board(
        BOARD_SIZE,
        Player("Black player", BLACK_COLOR),
        Player("White player", WHITE_COLOR),
    )

    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while not board.is_terminate():
        best, prob, value = pick_move(
            model=model, board=board, device=device, temperature=0.0
        )
        print(f"Move probability: {prob}")
        print(f"Value: {value}")

        if best is not None:
            board.place_move(best, BLACK_COLOR)
        else:
            board.pass_move()
        board.print_ascii_board()

        time.sleep(2.5)

        best, prob, value = pick_move(
            model=model, board=board, device=device, temperature=0.5
        )
        print(f"Move probability: {prob}")
        print(f"Value: {value}")

        if best is not None:
            board.place_move(best, WHITE_COLOR)
        else:
            board.pass_move()
        board.print_ascii_board()

        time.sleep(2.5)


if __name__ == "__main__":
    mcts_nn_vs_mcts_nn()
