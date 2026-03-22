"""
API tests for game_api endpoints.

These tests cover serializer validation, bounds checking, and view behaviour
without invoking the actual Go algorithms (which require a trained model file
and are tested elsewhere).

Django is configured by api/conftest.py; do not import Django or call
django.setup() here directly.
"""

from typing import Any
from unittest.mock import patch

import pytest
from common.exceptions import BadRequestError
from game_api.serializers import (AnalyzeParamsSerializer,
                                  AnalyzeRequestSerializer)
from game_api.services import _parse_move, _parse_player
from rest_framework.response import Response
from rest_framework.test import APIClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "board_size": 19,
        "rules": "japanese",
        "komi": 6.5,
        "to_play": "B",
        "moves": [],
        "algo": "minimax",
        "params": {"depth": 2},
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# AnalyzeRequestSerializer tests
# ---------------------------------------------------------------------------


class TestAnalyzeRequestSerializer:
    def test_valid_payload_passes(self) -> None:
        s = AnalyzeRequestSerializer(data=_valid_payload())
        assert s.is_valid(), s.errors

    def test_invalid_board_size_rejected(self) -> None:
        s = AnalyzeRequestSerializer(data=_valid_payload(board_size=7))
        assert not s.is_valid()
        assert "board_size" in s.errors

    def test_invalid_algo_rejected(self) -> None:
        s = AnalyzeRequestSerializer(data=_valid_payload(algo="alphabeta"))
        assert not s.is_valid()
        assert "algo" in s.errors

    def test_invalid_to_play_rejected(self) -> None:
        s = AnalyzeRequestSerializer(data=_valid_payload(to_play="X"))
        assert not s.is_valid()
        assert "to_play" in s.errors

    def test_valid_moves_normalised(self) -> None:
        payload = _valid_payload(moves=[["b", "d4"], ["W", "E5"]])
        s = AnalyzeRequestSerializer(data=payload)
        assert s.is_valid(), s.errors
        colors = [color for color, _ in s.validated_data["moves"]]
        assert colors == ["B", "W"]

    def test_move_invalid_color_rejected(self) -> None:
        s = AnalyzeRequestSerializer(data=_valid_payload(moves=[["X", "D4"]]))
        assert not s.is_valid()

    def test_move_empty_point_rejected(self) -> None:
        s = AnalyzeRequestSerializer(data=_valid_payload(moves=[["B", ""]]))
        assert not s.is_valid()


class TestAnalyzeParamsSerializer:
    def test_num_simulations_too_low_rejected(self) -> None:
        s = AnalyzeParamsSerializer(data={"num_simulations": 0})
        assert not s.is_valid()

    def test_num_simulations_too_high_rejected(self) -> None:
        s = AnalyzeParamsSerializer(data={"num_simulations": 5001})
        assert not s.is_valid()

    def test_num_simulations_valid(self) -> None:
        s = AnalyzeParamsSerializer(data={"num_simulations": 100})
        assert s.is_valid(), s.errors

    def test_depth_too_low_rejected(self) -> None:
        s = AnalyzeParamsSerializer(data={"depth": 0})
        assert not s.is_valid()

    def test_depth_too_high_rejected(self) -> None:
        s = AnalyzeParamsSerializer(data={"depth": 7})
        assert not s.is_valid()

    def test_depth_valid(self) -> None:
        s = AnalyzeParamsSerializer(data={"depth": 3})
        assert s.is_valid(), s.errors


# ---------------------------------------------------------------------------
# Service-layer unit tests
# ---------------------------------------------------------------------------


class TestParsePlayer:
    def test_parse_black(self) -> None:
        from lucidtree.constants import BLACK_COLOR

        assert _parse_player("B").get_color() == BLACK_COLOR

    def test_parse_white(self) -> None:
        from lucidtree.constants import WHITE_COLOR

        assert _parse_player("W").get_color() == WHITE_COLOR

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises(BadRequestError):
            _parse_player("X")


class TestParseMove:
    def test_parse_pass(self) -> None:
        from lucidtree.constants import PASS_MOVE_POSITION

        assert _parse_move("PASS", board_size=19) == PASS_MOVE_POSITION

    def test_parse_a1(self) -> None:
        assert _parse_move("A1", board_size=19) == (0, 0)

    def test_parse_j1_skips_i(self) -> None:
        # GTP skips 'I', so J → column 8
        assert _parse_move("J1", board_size=19) == (0, 8)

    def test_out_of_bounds_row_raises(self) -> None:
        with pytest.raises(BadRequestError):
            _parse_move("A20", board_size=19)  # row 19 is out of bounds

    def test_letter_i_rejected(self) -> None:
        with pytest.raises(BadRequestError):
            _parse_move("I1", board_size=19)

    def test_out_of_bounds_for_small_board(self) -> None:
        with pytest.raises(BadRequestError):
            _parse_move("K1", board_size=9)  # column 9 is out of bounds for 9x9


# ---------------------------------------------------------------------------
# Analyze service integration tests (algorithm mocked)
# ---------------------------------------------------------------------------


class TestAnalyzeService:
    @patch("lucidtree.engine.analysis.pick_move_minimax", return_value=(3, 3))
    def test_minimax_returns_gtp_move(self, _mock: Any) -> None:
        from game_api.services import analyze

        result = analyze(_valid_payload())
        assert "best_move" in result
        assert result["algorithm"] == "minimax"
        assert isinstance(result["best_move"], str)

    @patch("lucidtree.engine.analysis.pick_move_minimax", return_value=(3, 3))
    def test_stats_does_not_contain_raw_tuple(self, _mock: Any) -> None:
        from game_api.services import analyze

        result = analyze(_valid_payload())
        for v in result["stats"].values():
            assert not isinstance(v, tuple), f"Raw tuple found in stats: {v}"

    def test_mcts_on_9x9_raises(self) -> None:
        from game_api.services import analyze

        with pytest.raises((BadRequestError, ValueError)):
            analyze(_valid_payload(board_size=9, algo="mcts"))

    def test_nn_on_13x13_raises(self) -> None:
        from game_api.services import analyze

        with pytest.raises((BadRequestError, ValueError)):
            analyze(_valid_payload(board_size=13, algo="nn"))

    def test_invalid_move_raises_bad_request(self) -> None:
        from game_api.services import analyze

        with pytest.raises(BadRequestError):
            analyze(_valid_payload(moves=[["B", "Z99"]]))


# ---------------------------------------------------------------------------
# View-level tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client() -> APIClient:
    return APIClient()


class TestHealthView:
    def test_health_returns_200(self, api_client: APIClient) -> None:
        response = api_client.get("/api/health/")
        assert response.status_code == 200
        assert response.data["status"] == "ok"


class TestAnalyzeView:
    def _post(self, api_client: APIClient, payload: dict[str, Any]) -> Response:
        return api_client.post("/api/analyze/", payload, format="json")

    def test_invalid_board_size_returns_400(self, api_client: APIClient) -> None:
        response = self._post(api_client, {**_valid_payload(), "board_size": 7})
        assert response.status_code == 400

    def test_missing_algo_returns_400(self, api_client: APIClient) -> None:
        payload = {k: v for k, v in _valid_payload().items() if k != "algo"}
        response = self._post(api_client, payload)
        assert response.status_code == 400

    @patch(
        "game_api.services.analyze",
        return_value={
            "best_move": "D4",
            "algorithm": "minimax",
            "stats": {"depth": 1, "elapsed_ms": 5.0},
        },
    )
    def test_valid_request_returns_200(self, _mock: Any, api_client: APIClient) -> None:
        response = self._post(api_client, _valid_payload())
        assert response.status_code == 200
        assert "best_move" in response.data

    @patch("game_api.services.analyze", side_effect=RuntimeError("internal secret"))
    def test_unexpected_exception_returns_generic_500(
        self, _mock: Any, api_client: APIClient
    ) -> None:
        response = self._post(api_client, _valid_payload())
        assert response.status_code == 500
        # The raw exception message must NOT be exposed to the client
        assert "internal secret" not in str(response.data)
        assert "detail" in response.data
