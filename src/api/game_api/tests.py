# fmt: off

from typing import Any
from unittest.mock import patch

import pytest
from rest_framework.response import Response
from rest_framework.test import APIClient

from api.common.exceptions import BadRequestError
from api.game_api.serializers import AnalyzeRequestSerializer
from api.game_api.services import _parse_move, _parse_player

# fmt: on


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_params() -> dict[str, Any]:
    return {
        "num_simulations": 250,
        "c_puct": 1.7,
        "select_by": "value",
        "policy_weight": 1.2,
        "value_weight": 0.8,
        "dirichlet_alpha": 0.0,
        "dirichlet_epsilon": 0.0,
        "max_time_ms": 1000,
        "temperature": 0.0,
        "seed": 123,
    }


def _valid_output() -> dict[str, Any]:
    return {
        "include_top_moves": 5,
        "include_policy": False,
        "include_winrate": False,
        "include_visits": False,
    }


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rules": "japanese",
        "komi": 6.5,
        "to_play": "B",
        "moves": [],
        "algo": "mcts",
        "params": _valid_params(),
        "output": _valid_output(),
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

        assert _parse_move("PASS") == PASS_MOVE_POSITION

    def test_parse_a1(self) -> None:
        assert _parse_move("A1") == (0, 0)

    def test_parse_j1_skips_i(self) -> None:
        # GTP skips 'I', so J → column 8
        assert _parse_move("J1") == (0, 8)

    def test_out_of_bounds_row_raises(self) -> None:
        with pytest.raises(BadRequestError):
            _parse_move("A20")  # row 19 is out of bounds

    def test_letter_i_rejected(self) -> None:
        with pytest.raises(BadRequestError):
            _parse_move("I1")


# ---------------------------------------------------------------------------
# Analyze service integration tests (algorithm mocked)
# ---------------------------------------------------------------------------


class TestAnalyzeService:
    @patch(
        "lucidtree.engine.analysis.pick_moves_mcts",
        return_value=([(3, 3)], []),
    )
    def test_mcts_returns_gtp_move(self, _mock: Any) -> None:
        from api.game_api.services import analyze

        result = analyze(_valid_payload())
        assert "top_moves" in result
        assert result["algorithm"] == "mcts"
        assert isinstance(result["top_moves"], list)
        assert result["top_moves"] and isinstance(result["top_moves"][0], dict)
        assert result["top_moves"][0]["move"] == "D4"

    @patch(
        "lucidtree.engine.analysis.pick_moves_mcts",
        return_value=([(3, 3)], []),
    )
    def test_analysis_config_affects_mcts_call(self, mock_pick: Any) -> None:
        from api.game_api.services import analyze

        analyze(_valid_payload())

        mock_pick.assert_called_once()
        assert mock_pick.call_args.kwargs["num_simulations"] == 250
        assert mock_pick.call_args.kwargs["c_puct"] == 1.7
        assert mock_pick.call_args.kwargs["select_by"] == "value"
        assert "stats_out" in mock_pick.call_args.kwargs

    def test_invalid_move_raises_bad_request(self) -> None:
        from api.game_api.services import analyze

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

    def test_missing_algo_returns_400(self, api_client: APIClient) -> None:
        payload = {k: v for k, v in _valid_payload().items() if k != "algo"}
        response = self._post(api_client, payload)
        assert response.status_code == 400

    @patch(
        "api.game_api.services.analyze",
        return_value={
            "top_moves": ["D4"],
            "algorithm": "minimax",
            "stats": {"depth": 1, "elapsed_ms": 5.0},
        },
    )
    @pytest.mark.django_db
    def test_valid_request_returns_200(self, _mock: Any, api_client: APIClient) -> None:
        response = self._post(api_client, _valid_payload())
        assert response.status_code == 200
        assert "top_moves" in response.data

    @patch("api.game_api.services.analyze", side_effect=RuntimeError("internal secret"))
    @pytest.mark.django_db
    def test_unexpected_exception_returns_generic_500(
        self, _mock: Any, api_client: APIClient
    ) -> None:
        response = self._post(api_client, _valid_payload())
        assert response.status_code == 500
        assert "internal secret" not in str(response.data)
        assert "detail" in response.data
