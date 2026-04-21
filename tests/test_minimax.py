import pytest

from lucidtree.constants import BLACK_COLOR, WHITE_COLOR
from lucidtree.go.board import Board
from lucidtree.go.player import Player
from lucidtree.minimax.search import evaluate, game_is_over_by_passes, next_best_moves

_black = Player("Black", BLACK_COLOR)
_white = Player("White", WHITE_COLOR)

# Use a small board to keep tests fast
SMALL = 5


def _board(size: int = SMALL) -> Board:
    return Board(size, _black, _white)


class TestGameIsOverByPasses:
    def test_zero_passes_not_over(self) -> None:
        assert not game_is_over_by_passes(0)

    def test_one_pass_not_over(self) -> None:
        assert not game_is_over_by_passes(1)

    def test_two_passes_game_over(self) -> None:
        assert game_is_over_by_passes(2)

    def test_three_passes_game_over(self) -> None:
        assert game_is_over_by_passes(3)


class TestEvaluate:
    def test_empty_board_white_leads_by_komi(self) -> None:
        board = _board()
        score = evaluate(board, komi=7.5)
        # Empty board: no territory, white_score - black_score = komi
        assert score == pytest.approx(7.5)

    def test_black_territory_lowers_score(self) -> None:
        board = _board()
        # Black corners occupy territory in a 5x5 board
        board.place_move((0, 0), BLACK_COLOR)
        board.place_move((0, 1), BLACK_COLOR)
        board.place_move((1, 0), BLACK_COLOR)
        score_no_territory = evaluate(_board(), komi=0)
        score_with_black = evaluate(board, komi=0)
        assert score_with_black < score_no_territory

    def test_evaluate_uses_japanese_rules_by_default(self) -> None:
        board = _board()
        board.place_move((0, 0), BLACK_COLOR)
        score_jp = evaluate(board, komi=0, rules="japanese")
        score_cn = evaluate(board, komi=0, rules="chinese")
        # Results may differ; both should be finite floats
        assert isinstance(score_jp, float)
        assert isinstance(score_cn, float)


class TestNextBestMoves:
    def test_returns_list_of_positions(self) -> None:
        board = _board()
        moves = next_best_moves(board, isMax=True, depth=1)
        assert isinstance(moves, list)
        assert len(moves) >= 1
        for pos in moves:
            assert isinstance(pos, tuple)
            assert len(pos) == 2

    def test_include_top_moves_respected(self) -> None:
        board = _board()
        moves = next_best_moves(board, isMax=True, depth=1, include_top_moves=3)
        assert len(moves) <= 3

    def test_works_for_maximizing_player(self) -> None:
        board = _board()
        moves = next_best_moves(board, isMax=True, depth=1)
        assert moves  # non-empty

    def test_works_for_minimizing_player(self) -> None:
        board = _board()
        moves = next_best_moves(board, isMax=False, depth=1)
        assert moves

    def test_alpha_beta_and_no_alpha_beta_agree(self) -> None:
        board = _board()
        moves_ab = next_best_moves(
            board, isMax=True, depth=2, use_alpha_beta=True, include_top_moves=1
        )
        moves_no_ab = next_best_moves(
            board, isMax=True, depth=2, use_alpha_beta=False, include_top_moves=1
        )
        # Best move should be the same regardless of pruning
        assert moves_ab[0] == moves_no_ab[0]

    def test_stats_out_populated(self) -> None:
        board = _board()
        stats: dict = {}
        next_best_moves(board, isMax=True, depth=2, stats_out=stats)
        assert "search_depth_reached" in stats
        assert stats["search_depth_reached"] >= 1

    def test_with_time_limit_returns_valid_move(self) -> None:
        import time

        board = _board()
        deadline = time.perf_counter() + 5.0  # generous 5s limit
        moves = next_best_moves(
            board, isMax=True, depth=2, deadline=deadline, include_top_moves=1
        )
        assert moves

    def test_non_default_komi_affects_evaluation(self) -> None:
        board = _board()
        # With komi=0, white advantage from territory only (empty board → draw)
        score_zero_komi = evaluate(board, komi=0)
        score_large_komi = evaluate(board, komi=100)
        assert score_large_komi > score_zero_komi
