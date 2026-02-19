from typing import override

import pygame

from mini_katago.constants import BLACK_COLOR, WHITE_COLOR
from mini_katago.go.board import Board
from mini_katago.go.player import Player
from mini_katago.utils import get_project_root


class InteractiveBoard(Board):
    def __init__(self, size: int, black_player: Player, white_player: Player) -> None:
        super().__init__(size, black_player, white_player)

        # Constants for display
        self.MIN_WINDOW_SIZE = 600
        self.MAX_WINDOW_SIZE = 1200
        self.MARGIN = 60
        self.COORD_FONT_SIZE = 20

        # Colors
        self.GRID_COLOR = (0, 0, 0)
        self.BLACK_STONE_COLOR = (0, 0, 0)
        self.WHITE_STONE_COLOR = (255, 255, 255)
        self.BLACK_HOVER_COLOR = (50, 50, 50, 180)  # Semi-transparent dark gray
        self.WHITE_HOVER_COLOR = (200, 200, 200, 180)  # Semi-transparent light gray
        self.COORD_COLOR = (0, 0, 0)

        self.screen: pygame.Surface
        self.hover_pos: tuple[int, int] | None = None

    @override
    def show_board(self) -> None:
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

        # Calculate responsive window size
        self.base_size = max(
            self.MIN_WINDOW_SIZE, min(self.MAX_WINDOW_SIZE, self.size * 40)
        )
        self.board_area = self.base_size - 2 * self.MARGIN
        self.cell_size = self.board_area // self.size
        self.board_size = self.cell_size * (self.size - 1)
        self.window_size = self.board_size + 2 * self.MARGIN

        # Create window
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Go Board - Interactive")

        # Load background image
        # Get the project root directory (assuming board.py is in src/mini_katago/go/)
        root = get_project_root()
        bg_path = root / "assets" / "board-bg.png"

        if not bg_path.exists():
            raise FileNotFoundError(f"Background image not found at {bg_path}")

        self.bg_image = pygame.image.load(str(bg_path))
        self.bg_image = pygame.transform.scale(
            self.bg_image, (self.window_size, self.window_size)
        )

        # Font for coordinates
        self.font = pygame.font.Font(None, self.COORD_FONT_SIZE)

        # Main game loop
        self._running = True

        # Initial draw
        self.draw_board()

        # Start the game loop
        self.main_loop()

    def main_loop(self) -> None:
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False

                elif event.type == pygame.MOUSEMOTION:
                    # Update hover position
                    mouse_pos = pygame.mouse.get_pos()
                    board_pos = self.screen_to_board(mouse_pos)
                    if board_pos != self.hover_pos:
                        self.hover_pos = board_pos
                        self.draw_board()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_pos = pygame.mouse.get_pos()
                        board_pos = self.screen_to_board(mouse_pos)

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
                                    self.hover_pos = None
                                    self.draw_board()
                                except ValueError:
                                    # Invalid move, ignore
                                    pass

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._running = False

        pygame.quit()

    def get_column_label(self, col: int) -> str:
        """Convert column index to letter label (A-T, skipping I)"""
        # Skip 'I' (8th letter) as it's confusing with '1' in Go notation
        if col >= 8:
            return chr(ord("A") + col + 1)  # Skip I
        return chr(ord("A") + col)

    def get_row_label(self, row: int) -> str:
        """Convert row index to number label (1-19)"""
        return str(self.size - row)

    # Convert screen coordinates to board position
    def screen_to_board(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        """Convert screen coordinates to board (row, col) or None if outside board"""
        x, y = pos
        # Adjust for self.margin
        x -= self.MARGIN
        y -= self.MARGIN

        # Check if within board bounds (allow clicks up to border line)
        if (
            x < -self.cell_size // 2
            or y < -self.cell_size // 2
            or x > self.board_size + self.cell_size // 2
            or y > self.board_size + self.cell_size // 2
        ):
            return None

        # Find nearest intersection point
        # Round to nearest intersection
        col = round(x / self.cell_size)
        row = round(y / self.cell_size)

        # Clamp to valid range (0 to size-1)
        col = max(0, min(self.size - 1, col))
        row = max(0, min(self.size - 1, row))

        return (row, col)

    # Convert board position to screen coordinates
    def board_to_screen(self, row: int, col: int) -> tuple[int, int]:
        """Convert board (row, col) to screen coordinates (intersection point)"""
        x = self.MARGIN + col * self.cell_size
        y = self.MARGIN + row * self.cell_size
        return (x, y)

    # Draw the board
    def draw_board(self) -> None:
        """Draw the entire board"""
        # Draw background
        self.screen.blit(self.bg_image, (0, 0))

        # Draw grid lines
        for i in range(self.size):
            # Vertical lines
            start_x = self.MARGIN + i * self.cell_size
            pygame.draw.line(
                self.screen,
                self.GRID_COLOR,
                (start_x, self.MARGIN),
                (start_x, self.MARGIN + self.board_size),
                1,
            )
            # Horizontal lines
            start_y = self.MARGIN + i * self.cell_size
            pygame.draw.line(
                self.screen,
                self.GRID_COLOR,
                (self.MARGIN, start_y),
                (self.MARGIN + self.board_size, start_y),
                1,
            )

        # Draw the star points (if possible)
        if self.size == 19:
            star_points_locations = [3, 9, 15]
            for row in star_points_locations:
                for col in star_points_locations:
                    x, y = self.board_to_screen(row, col)
                    radius = int(self.cell_size * 0.2)
                    pygame.draw.circle(self.screen, self.GRID_COLOR, (x, y), radius)

        # Draw coordinates
        for i in range(self.size):
            # Column labels (top) - positioned at intersection points
            col_label = self.get_column_label(i)
            text_surface = self.font.render(col_label, True, self.COORD_COLOR)
            text_rect = text_surface.get_rect(
                center=(self.MARGIN + i * self.cell_size, self.MARGIN // 2)
            )
            self.screen.blit(text_surface, text_rect)

            # Column labels (bottom) - positioned at intersection points
            text_rect = text_surface.get_rect(
                center=(
                    self.MARGIN + i * self.cell_size,
                    self.window_size - self.MARGIN // 2,
                )
            )
            self.screen.blit(text_surface, text_rect)

            # Row labels (left) - positioned at intersection points
            row_label = self.get_row_label(i)
            text_surface = self.font.render(row_label, True, self.COORD_COLOR)
            text_rect = text_surface.get_rect(
                center=(self.MARGIN // 2, self.MARGIN + i * self.cell_size)
            )
            self.screen.blit(text_surface, text_rect)

            # Row labels (right) - positioned at intersection points
            text_rect = text_surface.get_rect(
                center=(
                    self.window_size - self.MARGIN // 2,
                    self.MARGIN + i * self.cell_size,
                )
            )
            self.screen.blit(text_surface, text_rect)

        # Draw stones
        for row in range(self.size):
            for col in range(self.size):
                move = self.get_move_at_position((row, col))
                if not move.is_empty():
                    x, y = self.board_to_screen(row, col)
                    color = (
                        self.BLACK_STONE_COLOR
                        if move.get_color() == BLACK_COLOR
                        else self.WHITE_STONE_COLOR
                    )
                    radius = int(self.cell_size * 0.4)
                    pygame.draw.circle(self.screen, color, (x, y), radius)
                    # Draw border for white stones
                    if move.get_color() == WHITE_COLOR:
                        pygame.draw.circle(
                            self.screen, self.GRID_COLOR, (x, y), radius, 1
                        )

        # Draw hover preview
        if self.hover_pos is not None:
            row, col = self.hover_pos
            move = self.get_move_at_position((row, col))

            # Only show hover if position is empty
            if move.is_empty():
                # Temporarily set color to check validity
                original_color = move.get_color()
                move.set_color(self.current_player.get_color())
                is_valid = self.move_is_valid(move)
                move.set_color(original_color)  # Restore original color

                if is_valid:
                    x, y = self.board_to_screen(row, col)
                    hover_color = (
                        self.BLACK_HOVER_COLOR
                        if self.current_player.get_color() == BLACK_COLOR
                        else self.WHITE_HOVER_COLOR
                    )
                    radius = int(self.cell_size * 0.35)

                    # Create a surface for semi-transparent circle
                    hover_surface = pygame.Surface(
                        (radius * 2, radius * 2), pygame.SRCALPHA
                    )
                    pygame.draw.circle(
                        hover_surface, hover_color, (radius, radius), radius
                    )
                    self.screen.blit(hover_surface, (x - radius, y - radius))

        # Update the display
        pygame.display.flip()
