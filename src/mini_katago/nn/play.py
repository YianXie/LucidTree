import time

import torch

from mini_katago.constants import BLACK_COLOR, BOARD_SIZE, WHITE_COLOR
from mini_katago.go.board import Board
from mini_katago.go.player import Player
from mini_katago.nn.agent import load_model, pick_move


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
    nn_vs_nn()
