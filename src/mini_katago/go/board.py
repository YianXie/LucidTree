# fmt: off
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pygame

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

        self._ko_positions = None

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
        black_territories = 0
        white_territories = 0
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

    def show_interactive_board(self) -> None:
        """
        Display a realistic and interactive game board with Pygame

        Features:
        - Responsive board size based on current board dimensions
        - Wood texture background
        - Grid lines and coordinate labels
        - Hover preview showing where the move will be placed
        - Hover color changes based on current player
        - No hover shown for invalid moves
        - Click to place moves automatically
        """
        # Initialize Pygame
        pygame.init()

        # Constants for display
        MIN_WINDOW_SIZE = 600
        MAX_WINDOW_SIZE = 1200
        MARGIN = 60  # Space for coordinates
        COORD_FONT_SIZE = 20

        # Calculate responsive window size
        base_size = max(MIN_WINDOW_SIZE, min(MAX_WINDOW_SIZE, self.size * 40))
        board_area = base_size - 2 * MARGIN
        cell_size = board_area // self.size
        board_size = cell_size * (self.size - 1)
        window_size = board_size + 2 * MARGIN

        # Create window
        screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Go Board - Interactive")

        # Load background image
        # Get the project root directory (assuming board.py is in src/mini_katago/go/)
        project_root = Path(__file__).parent.parent.parent.parent
        bg_path = project_root / "assets" / "board-bg.png"

        if not bg_path.exists():
            raise FileNotFoundError(f"Background image not found at {bg_path}")

        bg_image = pygame.image.load(str(bg_path))
        bg_image = pygame.transform.scale(bg_image, (window_size, window_size))

        # Colors
        GRID_COLOR = (0, 0, 0)
        BLACK_STONE_COLOR = (0, 0, 0)
        WHITE_STONE_COLOR = (255, 255, 255)
        BLACK_HOVER_COLOR = (50, 50, 50, 180)  # Semi-transparent dark gray
        WHITE_HOVER_COLOR = (200, 200, 200, 180)  # Semi-transparent light gray
        COORD_COLOR = (0, 0, 0)

        # Font for coordinates
        font = pygame.font.Font(None, COORD_FONT_SIZE)

        # Coordinate labels (A-T for columns, skipping I, 1-19 for rows)
        def get_column_label(col: int) -> str:
            """Convert column index to letter label (A-T, skipping I)"""
            # Skip 'I' (8th letter) as it's confusing with '1' in Go notation
            if col >= 8:
                return chr(ord("A") + col + 1)  # Skip I
            return chr(ord("A") + col)

        def get_row_label(row: int) -> str:
            """Convert row index to number label (1-19)"""
            return str(self.size - row)

        # Convert screen coordinates to board position
        def screen_to_board(pos: tuple[int, int]) -> tuple[int, int] | None:
            """Convert screen coordinates to board (row, col) or None if outside board"""
            x, y = pos
            # Adjust for margin
            x -= MARGIN
            y -= MARGIN

            # Check if within board bounds (allow clicks up to border line)
            if (
                x < -cell_size // 2
                or y < -cell_size // 2
                or x > board_size + cell_size // 2
                or y > board_size + cell_size // 2
            ):
                return None

            # Find nearest intersection point
            # Round to nearest intersection
            col = round(x / cell_size)
            row = round(y / cell_size)

            # Clamp to valid range (0 to size-1)
            col = max(0, min(self.size - 1, col))
            row = max(0, min(self.size - 1, row))

            return (row, col)

        # Convert board position to screen coordinates
        def board_to_screen(row: int, col: int) -> tuple[int, int]:
            """Convert board (row, col) to screen coordinates (intersection point)"""
            x = MARGIN + col * cell_size
            y = MARGIN + row * cell_size
            return (x, y)

        # Draw the board
        def draw_board(hover_pos: tuple[int, int] | None = None) -> None:
            """Draw the entire board"""
            # Draw background
            screen.blit(bg_image, (0, 0))

            # Draw grid lines
            for i in range(self.size):
                # Vertical lines
                start_x = MARGIN + i * cell_size
                pygame.draw.line(
                    screen,
                    GRID_COLOR,
                    (start_x, MARGIN),
                    (start_x, MARGIN + board_size),
                    1,
                )
                # Horizontal lines
                start_y = MARGIN + i * cell_size
                pygame.draw.line(
                    screen,
                    GRID_COLOR,
                    (MARGIN, start_y),
                    (MARGIN + board_size, start_y),
                    1,
                )

            # Draw the star points (if possible)
            if self.size == 19:
                star_points_locations = [3, 9, 15]
                for row in star_points_locations:
                    for col in star_points_locations:
                        x, y = board_to_screen(row, col)
                        radius = int(cell_size * 0.2)
                        pygame.draw.circle(screen, GRID_COLOR, (x, y), radius)

            # Draw coordinates
            for i in range(self.size):
                # Column labels (top) - positioned at intersection points
                col_label = get_column_label(i)
                text_surface = font.render(col_label, True, COORD_COLOR)
                text_rect = text_surface.get_rect(
                    center=(MARGIN + i * cell_size, MARGIN // 2)
                )
                screen.blit(text_surface, text_rect)

                # Column labels (bottom) - positioned at intersection points
                text_rect = text_surface.get_rect(
                    center=(
                        MARGIN + i * cell_size,
                        window_size - MARGIN // 2,
                    )
                )
                screen.blit(text_surface, text_rect)

                # Row labels (left) - positioned at intersection points
                row_label = get_row_label(i)
                text_surface = font.render(row_label, True, COORD_COLOR)
                text_rect = text_surface.get_rect(
                    center=(MARGIN // 2, MARGIN + i * cell_size)
                )
                screen.blit(text_surface, text_rect)

                # Row labels (right) - positioned at intersection points
                text_rect = text_surface.get_rect(
                    center=(
                        window_size - MARGIN // 2,
                        MARGIN + i * cell_size,
                    )
                )
                screen.blit(text_surface, text_rect)

            # Draw stones
            for row in range(self.size):
                for col in range(self.size):
                    move = self.get_move_at_position((row, col))
                    if not move.is_empty():
                        x, y = board_to_screen(row, col)
                        color = (
                            BLACK_STONE_COLOR
                            if move.get_color() == BLACK_COLOR
                            else WHITE_STONE_COLOR
                        )
                        radius = int(cell_size * 0.4)
                        pygame.draw.circle(screen, color, (x, y), radius)
                        # Draw border for white stones
                        if move.get_color() == WHITE_COLOR:
                            pygame.draw.circle(screen, GRID_COLOR, (x, y), radius, 1)

            # Draw hover preview
            if hover_pos is not None:
                row, col = hover_pos
                move = self.get_move_at_position((row, col))

                # Only show hover if position is empty
                if move.is_empty():
                    # Temporarily set color to check validity
                    original_color = move.get_color()
                    move.set_color(self.current_player.get_color())
                    is_valid = self.move_is_valid(move)
                    move.set_color(original_color)  # Restore original color

                    if is_valid:
                        x, y = board_to_screen(row, col)
                        hover_color = (
                            BLACK_HOVER_COLOR
                            if self.current_player.get_color() == BLACK_COLOR
                            else WHITE_HOVER_COLOR
                        )
                        radius = int(cell_size * 0.35)

                        # Create a surface for semi-transparent circle
                        hover_surface = pygame.Surface(
                            (radius * 2, radius * 2), pygame.SRCALPHA
                        )
                        pygame.draw.circle(
                            hover_surface, hover_color, (radius, radius), radius
                        )
                        screen.blit(hover_surface, (x - radius, y - radius))

            pygame.display.flip()

        # Main game loop
        running = True
        hover_pos: tuple[int, int] | None = None

        # Initial draw
        draw_board()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEMOTION:
                    # Update hover position
                    mouse_pos = pygame.mouse.get_pos()
                    board_pos = screen_to_board(mouse_pos)
                    if board_pos != hover_pos:
                        hover_pos = board_pos
                        draw_board(hover_pos)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_pos = pygame.mouse.get_pos()
                        board_pos = screen_to_board(mouse_pos)

                        if board_pos is not None:
                            row, col = board_pos
                            move = self.get_move_at_position((row, col))

                            # Only place if position is empty
                            if move.is_empty():
                                try:
                                    # Place the move with current player's color
                                    self.place_move(
                                        (row, col), self.current_player.get_color()
                                    )
                                    # Redraw board without hover
                                    hover_pos = None
                                    draw_board()
                                except ValueError:
                                    # Invalid move, ignore
                                    pass

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

        pygame.quit()

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
