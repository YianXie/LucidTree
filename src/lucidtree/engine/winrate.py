import torch

from api.common.exceptions import BadRequestError
from api.common.utils import parse_move, parse_player
from lucidtree.constants import BOARD_SIZE, PASS_MOVE_POSITION
from lucidtree.go.board import Board
from lucidtree.go.player import Player
from lucidtree.nn.agent import get_policy_value, load_model
from lucidtree.nn.features import value_to_winrate
from lucidtree.nn.model import PolicyValueNetwork


def generate_winrate(
    moves: list[list[str]],
    *,
    device: torch.device,
    temperature: float,
    model: PolicyValueNetwork | None = None,
) -> list[dict[str, float]]:
    """
    Generate the winrate based on the given board

    Args:
        board (Board): the board to generate winrate
        device (torch.device): the device to use (e.g., cpu)
        temperature (float): the temperature to apply to the neural network
        model (PolicyValueNetwork | None): the nn model, initialize one if not given

    Returns:
        dict[str, float]: the winrate in the format {black: x, white: x}
    """
    if model is None or not isinstance(model, PolicyValueNetwork):
        model = load_model()
    board = Board(BOARD_SIZE, Player.black(), Player.white())

    winrates = []
    for color_text, points_text in moves:
        player = parse_player(color_text)
        move_position = parse_move(points_text)

        try:
            if move_position == PASS_MOVE_POSITION:
                board.pass_move()
            else:
                board.place_move(move_position, player.get_color())
        except Exception as e:
            raise BadRequestError(f"Invalid move: {e}")

        _, value = get_policy_value(
            model, board, device=device, temperature=temperature
        )
        winrates.append(value_to_winrate(value, board.get_current_player().get_color()))

    return winrates
