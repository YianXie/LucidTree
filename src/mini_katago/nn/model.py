from mini_katago.go.board import Board


class PVNet:
    """
    Policy and value network
    """

    @staticmethod
    def predict(board: Board) -> tuple[list[float], float]:
        """
        Returns the policy and value of a given board state

        Args:
            board (Board): the board state

        Returns:
            tuple[list[float], float]: the policy (probability for each move) and the value (win rate)
        """
        return [0.5, 0, 5], 0.5
