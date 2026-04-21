from typing import Any
from unittest.mock import patch

import pytest

from lucidtree.constants import BOARD_SIZE
from lucidtree.go.board import Board
from lucidtree.go.exceptions import BadRequestError
from lucidtree.go.player import Player

_black = Player.black()
_white = Player.white()

_black.opponent = _white
_white.opponent = _black


def _make_board(size: int = BOARD_SIZE) -> Board:
    b = Board(size, Player.black(), Player.white())
    return b


def _minimax_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "depth": 1,
        "use_alpha_beta": True,
        "max_time_ms": None,
    }
    params.update(overrides)
    return params


def _output(**overrides: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "include_top_moves": 1,
        "include_policy": False,
        "include_winrate": False,
        "include_visits": False,
    }
    out.update(overrides)
    return out


# ---------------------------------------------------------------------------
# Helper: build players with opponent set (required by MCTS / analysis)
# ---------------------------------------------------------------------------


def _to_play_black() -> Player:
    black = Player.black()
    white = Player.white()
    black.opponent = white
    white.opponent = black
    return black


# ---------------------------------------------------------------------------
# Wrong board size
# ---------------------------------------------------------------------------


class TestAnalyzePositionBoardSizeValidation:
    def test_non_19x19_raises_bad_request(self) -> None:
        from lucidtree.engine.analysis import analyze_position

        board = _make_board(size=9)
        black = _to_play_black()
        with pytest.raises(BadRequestError, match="19x19"):
            analyze_position(
                algo="mcts",
                board=board,
                komi=7.5,
                rules="japanese",
                to_play=black,
                params=_minimax_params(),
                output=_output(),
            )

    def test_minimax_with_non_19x19_raises(self) -> None:
        """analyze_position enforces 19x19 for all algos."""
        from lucidtree.engine.analysis import analyze_position

        board = _make_board(size=9)
        black = _to_play_black()
        with pytest.raises(BadRequestError):
            analyze_position(
                algo="minimax",
                board=board,
                komi=7.5,
                rules="japanese",
                to_play=black,
                params=_minimax_params(),
                output=_output(),
            )


# ---------------------------------------------------------------------------
# Invalid algorithm
# ---------------------------------------------------------------------------


class TestAnalyzePositionInvalidAlgo:
    def test_invalid_algo_raises_bad_request(self) -> None:
        from lucidtree.engine.analysis import analyze_position

        board = _make_board()
        black = _to_play_black()
        with pytest.raises(BadRequestError, match="Invalid algorithm"):
            analyze_position(
                algo="alphabeta",
                board=board,
                komi=7.5,
                rules="japanese",
                to_play=black,
                params=_minimax_params(),
                output=_output(),
            )


# ---------------------------------------------------------------------------
# Negative max_time_ms
# ---------------------------------------------------------------------------


class TestAnalyzePositionMaxTimeMs:
    def test_negative_max_time_ms_raises(self) -> None:
        from lucidtree.engine.analysis import analyze_position

        board = _make_board()
        black = _to_play_black()
        with pytest.raises(BadRequestError, match="max_time_ms"):
            analyze_position(
                algo="minimax",
                board=board,
                komi=7.5,
                rules="japanese",
                to_play=black,
                params=_minimax_params(max_time_ms=-1),
                output=_output(),
            )


# ---------------------------------------------------------------------------
# Invalid include_top_moves
# ---------------------------------------------------------------------------


class TestAnalyzePositionIncludeTopMoves:
    def test_zero_include_top_moves_raises(self) -> None:
        from lucidtree.engine.analysis import analyze_position

        board = _make_board()
        black = _to_play_black()
        with pytest.raises(BadRequestError, match="include_top_moves"):
            analyze_position(
                algo="minimax",
                board=board,
                komi=7.5,
                rules="japanese",
                to_play=black,
                params=_minimax_params(),
                output=_output(include_top_moves=0),
            )

    def test_non_int_include_top_moves_raises(self) -> None:
        from lucidtree.engine.analysis import analyze_position

        board = _make_board()
        black = _to_play_black()
        with pytest.raises(BadRequestError, match="include_top_moves"):
            analyze_position(
                algo="minimax",
                board=board,
                komi=7.5,
                rules="japanese",
                to_play=black,
                params=_minimax_params(),
                output=_output(include_top_moves="five"),
            )


# ---------------------------------------------------------------------------
# Minimax integration (no model file needed)
# ---------------------------------------------------------------------------


class TestAnalyzePositionMinimax:
    @patch(
        "lucidtree.engine.analysis.pick_moves_minimax",
        return_value=[(3, 3)],
    )
    def test_minimax_returns_expected_shape(self, _mock: Any) -> None:
        from lucidtree.engine.analysis import analyze_position

        board = _make_board()
        black = _to_play_black()
        result = analyze_position(
            algo="minimax",
            board=board,
            komi=7.5,
            rules="japanese",
            to_play=black,
            params=_minimax_params(),
            output=_output(),
        )
        assert result["algorithm"] == "minimax"
        assert "top_moves" in result
        assert "stats" in result
        assert result["top_moves"][0]["move"] == "D4"

    @patch(
        "lucidtree.engine.analysis.pick_moves_minimax",
        return_value=[(3, 3), (4, 4)],
    )
    def test_minimax_include_top_moves(self, _mock: Any) -> None:
        from lucidtree.engine.analysis import analyze_position

        board = _make_board()
        black = _to_play_black()
        result = analyze_position(
            algo="minimax",
            board=board,
            komi=7.5,
            rules="japanese",
            to_play=black,
            params=_minimax_params(),
            output=_output(include_top_moves=2),
        )
        assert len(result["top_moves"]) == 2

    @patch(
        "lucidtree.engine.analysis.pick_moves_minimax",
        return_value=[(-1, -1)],  # PASS
    )
    def test_minimax_pass_move_in_result(self, _mock: Any) -> None:
        from lucidtree.engine.analysis import analyze_position

        board = _make_board()
        black = _to_play_black()
        result = analyze_position(
            algo="minimax",
            board=board,
            komi=7.5,
            rules="japanese",
            to_play=black,
            params=_minimax_params(),
            output=_output(),
        )
        assert result["top_moves"][0]["move"] == "PASS"

    @patch(
        "lucidtree.engine.analysis.pick_moves_minimax",
        return_value=[(3, 3)],
    )
    def test_elapsed_ms_in_stats(self, _mock: Any) -> None:
        from lucidtree.engine.analysis import analyze_position

        board = _make_board()
        black = _to_play_black()
        result = analyze_position(
            algo="minimax",
            board=board,
            komi=7.5,
            rules="japanese",
            to_play=black,
            params=_minimax_params(),
            output=_output(),
        )
        assert "elapsed_ms" in result["stats"]
        assert isinstance(result["stats"]["elapsed_ms"], float)
