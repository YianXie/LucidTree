# fmt: off
from collections import deque
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from mini_katago.constants import (BLACK_COLOR, EMPTY_COLOR,
                                   PASS_MOVE_POSITION, WHITE_COLOR)
from mini_katago.go.move import Move
from mini_katago.go.player import Player
from mini_katago.go.rules import Rules

# fmt: on


class Board:
    """
    A class representing a Go board

    Colors:
        -1: Black
        0: Empty
        1: White
    """

    def __init__(
        self,
        size: int,
        black_player: Player,
        white_player: Player,
    ) -> None:
        """
        Initialize a Go board

        Args:
            size (int): the size of the board
            black_player (Player): the black player
            white_player (Player): the white player
        """
        self.size: int = size
        self.black_player: Player = black_player
        self.white_player: Player = white_player
        self.current_player: Player = black_player
        self.state: list[list[Move]] = [
            [Move(row, col) for col in range(size)] for row in range(size)
        ]
        self._ko_positions: tuple[int, int] | None = None
        self._consecutive_passes: int = 0
        self._is_terminate: bool = False
        self._move_history: list[dict[str, Any]] = []

    def get_current_player(self) -> Player:
        """
        Get the current playing player

        Returns:
            Player: the player that is currently playing
        """
        return self.current_player

    def get_black_player(self) -> Player:
        """
        Get the black player

        Returns:
            Player: the black player
        """
        return self.black_player

    def get_white_player(self) -> Player:
        """
        Get the white player

        Returns:
            Player: the white player
        """
        return self.white_player

    def get_size(self) -> int:
        """
        Get the board size

        Returns:
            int: the board size
        """
        return self.size

    def get_nth_move(self, index: int) -> Move | None:
        """
        Get the nth move since game started. Negative index are also supported.

        Args:
            index (int): the index of the move

        Returns:
            Move | None: the nth move, or None if the index is invalid or if there are no moves made yet
        """
        if not self._move_history:
            return None

        if index < -len(self._move_history) or index >= len(self._move_history):
            return None

        move_info = self._move_history[index]

        # Check if the move is a pass
        if move_info["type"] == "pass":
            return Move(passed=True)

        row, col = move_info["position"]
        return Move(row, col, move_info["color"])

    def get_last_move(self) -> Move | None:
        """
        Get the last move in the current game

        Returns:
            Move | None: the last move, or None if the board is empty
        """
        return self.get_nth_move(-1)

    def get_all_moves(self) -> list[Move]:
        """
        Get all the moves on the board

        Returns:
            list[Move]: all the moves
        """
        if not self._move_history:
            return []

        moves: list[Move] = []
        for move in self._move_history:
            if move["type"] == "pass":
                moves.append(Move(passed=True))
            else:
                row, col = move["position"]
                moves.append(Move(row, col, move["color"]))

        return moves

    def get_move_at_position(self, position: tuple[int, int]) -> Move:
        """
        Get the move at the given position

        Args:
            position (tuple): the position of the move

        Returns:
            Move: the move at the given position
        """
        if not Rules.position_is_valid(position, self.size):
            raise ValueError(f"Invalid position: {position}")
        return self.state[position[0]][position[1]]

    def get_neighbors(self, move: Move) -> list[Move] | None:
        """
        Get the neighbors of a given position (maximum 4, minimum 2)

        Args:
            move (Move): the move

        Returns:
            list: a list of the neighbors of the given position
        """
        position = move.get_position()
        if position == PASS_MOVE_POSITION:
            return None

        neighbors = []
        if position[0] - 1 >= 0:
            neighbors.append(self.get_move_at_position((position[0] - 1, position[1])))
        if position[0] + 1 < self.size:
            neighbors.append(self.get_move_at_position((position[0] + 1, position[1])))
        if position[1] - 1 >= 0:
            neighbors.append(self.get_move_at_position((position[0], position[1] - 1)))
        if position[1] + 1 < self.size:
            neighbors.append(self.get_move_at_position((position[0], position[1] + 1)))
        return neighbors

    def get_ko_point(self) -> tuple[int, int] | None:
        """
        Get the current ko position (if any)

        Returns:
            tuple[int, int] | None: the current tuple position, None if does not exist
        """
        return self._ko_positions

    def get_connected(self, move: Move) -> list[Move]:
        """
        Count how many moves are connected to the given move (including the given move)

        Args:
            move (Move): the move to start counting from

        Returns:
            list: a list of all the connected moves with the same color of the given move
        """
        queue = deque[Move]([move])
        visited = set[Move]([move])
        connected = list[Move]([move])
        while queue:
            queued_move = queue.popleft()
            if queued_move.is_passed():
                continue
            neighbors = self.get_neighbors(queued_move)
            for neighbor in neighbors:  # type: ignore
                if neighbor not in visited and neighbor.get_color() == move.get_color():
                    queue.append(neighbor)
                    connected.append(neighbor)
                    visited.add(neighbor)
        return connected

    def get_legal_moves(self, color: int) -> list[Move]:
        """
        Get all legal moves for a given player

        Args:
            color (int): the color of the player to get all legal moves with

        Returns:
            list[Move]: all legal moves for the given player
        """
        moves: list[Move] = []
        for row in self.state:
            for move in row:
                if not move.is_empty():
                    continue
                # Test validity by temporarily setting color
                move.set_color(color)
                is_valid = self.move_is_valid(move)
                move.set_color(EMPTY_COLOR)  # Restore to empty
                if is_valid:
                    moves.append(move)
        return moves + [Move(passed=True)]

    def is_terminate(self) -> bool:
        """
        Check if the game is over

        Returns:
            bool: True if the game is over, False otherwise
        """
        return self._is_terminate

    def count_liberties(self, move: Move) -> int:
        """
        Iterative solution to count the liberties of a given position

        Args:
            move (Move): the move

        Returns:
            int: the amount of liberties of that position, -1 if move is empty
        """
        color = move.get_color()
        if color == EMPTY_COLOR:
            return -1

        liberties = 0
        queue = deque[Move]([move])
        visited = set[Move]([move])
        while queue:
            queuedMove = queue.popleft()
            if queuedMove.is_passed():
                continue
            neighbors = self.get_neighbors(queuedMove)
            for neighbor in neighbors:  # type: ignore
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                neighborColor = neighbor.get_color()
                if neighborColor == color:
                    queue.append(neighbor)
                elif neighborColor == EMPTY_COLOR:
                    liberties += 1

        return liberties

    def undo(self) -> None:
        """
        Undo the last move played by any player.

        This function restores the board to the state before the last move was played,
        including restoring captured stones and switching back the player.

        Raises:
            RuntimeError: if there are no moves to undo (game just started)
        """
        # Check if there's any move history to undo
        if not hasattr(self, "_move_history") or len(self._move_history) == 0:
            raise RuntimeError("No moves to undo: the game has just started!")

        # Pop the last move from history
        last_move_info = self._move_history.pop()

        # Extract information from the last move
        move_type = last_move_info["type"]  # either "place" or "pass"
        position = last_move_info["position"]
        color = last_move_info["color"]
        captures = last_move_info["captures"]
        previous_ko = last_move_info["previous_ko"]
        previous_consecutive_passes = last_move_info["previous_consecutive_passes"]
        previous_is_terminate = last_move_info["previous_is_terminate"]
        previous_capture_count = last_move_info["previous_capture_count"]

        if move_type == "place":
            # Remove the stone from the board
            row, col = position
            self.state[row][col].set_color(EMPTY_COLOR)

            # Restore captured stones
            for capture_position, capture_color in captures:
                captured_row, captured_col = capture_position
                self.state[captured_row][captured_col].set_color(capture_color)

            # Restore the capture count of the player who made the move
            if color == BLACK_COLOR:  # Black player made the move
                self.black_player.set_capture_count(previous_capture_count)
            else:  # White player made the move
                self.white_player.set_capture_count(previous_capture_count)

        # Restore Ko position
        self._ko_positions = previous_ko

        # Restore consecutive passes counter
        self._consecutive_passes = previous_consecutive_passes

        # Restore game termination state
        self._is_terminate = previous_is_terminate

        # Switch back to the previous player
        self.current_player = (
            self.black_player
            if self.current_player is self.white_player
            else self.white_player
        )

    def move_is_valid(self, move: Move) -> bool:
        """
        Check if a given move is valid

        Args:
            move (Move): the move to check

        Returns:
            bool: True if move is valid, False otherwise
        """
        # Prevent suicide
        if not self.check_captures(move) and self.count_liberties(move) <= 0:
            return False

        # Prevent playing the Ko directly after
        elif move.get_position() == self._ko_positions:
            return False

        return True

    def check_captures(self, move: Move) -> list[Move]:
        """
        Check if a given move captured any stones

        Args:
            move (Move): the move to check

        Returns:
            list[Move]: the stones that are captured, or empty list if none
        """
        if move.is_passed():
            return []
        captures = []
        seen = set[Move]()
        for neighbor in self.get_neighbors(move):  # type: ignore
            if neighbor.get_color() == move.get_color() * -1 and neighbor not in seen:
                if self.count_liberties(neighbor) == 0:
                    group = self.get_connected(neighbor)
                    captures.extend(group)
                    seen.update(group)

        return captures

    def place_move(self, position: tuple[int, int], color: int) -> None:
        """
        Place a move on the board

        Args:
            position (tuple): the position of the move
            color (int): the color of the move

        Raises:
            ValueError: if the position is invalid
            ValueError: if the color is invalid
            ValueError: if the position is already occupied
        """
        if not Rules.position_is_valid(position, self.size):
            raise ValueError(f"Invalid position: {position}")
        if not Rules.color_is_valid(color):
            raise ValueError(f"Invalid color: {color}")
        if not self.get_move_at_position(position).is_empty():
            raise ValueError(f"Position already occupied: {position}")
        if self._is_terminate:
            raise RuntimeError("Game is already over!")

        move: Move = self.state[position[0]][position[1]]
        prev_color = move.get_color()
        move.set_color(color)
        if not self.move_is_valid(move):
            move.set_color(prev_color)
            raise ValueError("Illegal move")

        # Calculate captures
        captures: list[Move] = self.check_captures(move)
        # Store capture positions and colors as tuples instead of deep copying Move objects
        capture_info: list[tuple[tuple[int, int], int]] = [
            (capture.get_position(), capture.get_color()) for capture in captures
        ]
        for capture in captures:
            row, col = capture.get_position()
            self.state[row][col].set_color(EMPTY_COLOR)

        self._move_history.append(
            {
                "type": "place",
                "position": position,
                "color": color,
                "captures": capture_info,
                "previous_ko": self._ko_positions,
                "previous_consecutive_passes": self._consecutive_passes,
                "previous_is_terminate": self._is_terminate,
                "previous_capture_count": self.current_player.get_capture_count(),
            }
        )

        # Increase the capture count after saving it to the history
        self.current_player.increase_capture_count(len(captures))

        # Clear the previous Ko
        self._ko_positions = None

        # Check for Ko
        if (
            len(captures) == 1
            and len(self.get_connected(move)) == 1
            and self.count_liberties(move) == 1
        ):
            self._ko_positions = captures[0].get_position()

        # Switch the player
        self.current_player = (
            self.black_player
            if self.current_player is self.white_player
            else self.white_player
        )

        # Reset the consecutive passes counter
        self._consecutive_passes = 0

    def pass_move(self) -> None:
        """
        Make a player passes a move
        """
        # Handle edge case — if the game is already over
        if self._is_terminate:
            raise RuntimeError("Game is already over!")

        # Append to move history
        self._move_history.append(
            {
                "type": "pass",
                "position": None,
                "color": self.current_player.get_color(),
                "captures": [],
                "previous_ko": self._ko_positions,
                "previous_consecutive_passes": self._consecutive_passes,
                "previous_is_terminate": self._is_terminate,
                "previous_capture_count": None,
            }
        )

        # Switches the current player
        self.current_player = (
            self.black_player
            if self.current_player is self.white_player
            else self.white_player
        )

        # Increase the counter
        self._consecutive_passes += 1
        if self._consecutive_passes >= 2:
            self._is_terminate = True

    def calculate_score(self) -> tuple[int, int]:
        """
        Estimate the territories for black and white player

        Returns:
            tuple: a tuple containing the territories for both side in the format (black, white)
        """
        visited = set[Move]()
        black_territories = white_territories = 0
        for row in self.state:
            for move in row:
                if move in visited:
                    continue
                if move.is_empty():
                    queue = deque[Move]([move])
                    queue_visited = set[Move]([move])
                    queued_neighbor_border_colors = set[int]()
                    empty_moves = 1  # include the move itself
                    while queue:
                        queuedMove = queue.popleft()
                        if queuedMove.is_passed():
                            continue
                        neighbors = self.get_neighbors(queuedMove)
                        for neighbor in neighbors:  # type: ignore
                            if neighbor in queue_visited:
                                continue
                            queue_visited.add(neighbor)
                            if not neighbor.is_empty():
                                queued_neighbor_border_colors.add(neighbor.get_color())
                            else:
                                empty_moves += 1
                                queue.append(neighbor)
                    if (
                        BLACK_COLOR in queued_neighbor_border_colors
                        and WHITE_COLOR not in queued_neighbor_border_colors
                    ):
                        black_territories += empty_moves
                    elif (
                        BLACK_COLOR not in queued_neighbor_border_colors
                        and WHITE_COLOR in queued_neighbor_border_colors
                    ):
                        white_territories += empty_moves
                    visited.update(queue_visited)
                visited.add(move)

        return (black_territories, white_territories)

    def print_ascii_board(self) -> None:
        """
        Print the ascii board in the terminal
        """
        print()
        for row in self.state:
            for move in row:
                color = move.get_color()
                print(
                    "B"
                    if color == BLACK_COLOR
                    else "W"
                    if color == WHITE_COLOR
                    else ".",
                    end=" ",
                )
            print()
        print()

    def show_board(self) -> None:
        """
        Display the board
        """
        fig = plt.figure(figsize=[9, 9])
        fig.patch.set_facecolor((0.85, 0.64, 0.125))
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        for x in range(self.size):
            ax.plot([x, x], [0, self.size - 1], "k")
        for y in range(self.size):
            ax.plot([0, self.size - 1], [y, y], "k")
        ax.set_position((0.0, 0.0, 1.0, 1.0))

        for row in self.state:
            for move in row:
                if move.is_empty():
                    continue
                moveRow, moveCol = move.get_position()
                color = (
                    "black"
                    if move.get_color() == BLACK_COLOR
                    else "white"
                    if move.get_color() == WHITE_COLOR
                    else "empty"
                )

                circle = patches.Circle(
                    (moveCol, self.size - moveRow - 1),
                    radius=0.4,
                    color=color,
                    zorder=3,
                )
                ax.add_patch(circle)

        ax.set_aspect("equal", adjustable="box")
        plt.show()

    def __eq__(self, other: object, /) -> bool:
        """
        Method to compare two Board object

        Args:
            other (object): another Board object

        Returns:
            bool: true if two board objects are equal, false otherwise
        """
        if not isinstance(other, Board):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        """
        Return a developer-friendly message for debugging

        Returns:
            str: the message
        """
        return f"===Board Information===\nBoard size: {self.size}\nBlack player: {self.black_player!r}\nWhite player: {self.white_player!r}"
