"""Regression: policy softmax temperature must fall back to general.temperature."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from lucidtree.constants import BLACK_COLOR, BOARD_SIZE, WHITE_COLOR
from lucidtree.engine.analysis import analyze_position
from lucidtree.go.board import Board
from lucidtree.go.player import Player

_test_black = Player("B", BLACK_COLOR)
_test_white = Player("W", WHITE_COLOR)


def _empty_board() -> Board:
    return Board(BOARD_SIZE, _test_black, _test_white)


def _nn_config_base() -> dict[str, Any]:
    return {
        "general": {
            "algorithm": "nn",
            "rules": "japanese",
            "komi": 6.5,
            "max_time_ms": 0,
            "temperature": 0.75,
            "seed": 1,
        },
        "neural_network": {
            "model": "checkpoint_19x19",
            "use_value_head": True,
        },
    }


@patch("lucidtree.engine.analysis.load_model", return_value=MagicMock())
@patch(
    "lucidtree.engine.analysis.pick_move_nn",
    return_value=((3, 3), 0.5, 0.1),
)
def test_nn_uses_general_temperature_when_policy_key_missing(
    mock_pick: MagicMock, _load: MagicMock
) -> None:
    cfg = _nn_config_base()
    analyze_position(
        _empty_board(),
        _test_black,
        "nn",
        cfg,
    )
    assert mock_pick.call_args.kwargs["temperature"] == 0.75


@patch("lucidtree.engine.analysis.load_model", return_value=MagicMock())
@patch(
    "lucidtree.engine.analysis.pick_move_nn",
    return_value=((3, 3), 0.5, 0.1),
)
def test_nn_policy_softmax_temperature_overrides_general(
    mock_pick: MagicMock, _load: MagicMock
) -> None:
    cfg = _nn_config_base()
    cfg["neural_network"]["policy_softmax_temperature"] = 0.2
    analyze_position(
        _empty_board(),
        _test_black,
        "nn",
        cfg,
    )
    assert mock_pick.call_args.kwargs["temperature"] == 0.2


@patch("lucidtree.engine.analysis.load_model", return_value=MagicMock())
@patch(
    "lucidtree.engine.analysis.pick_move_nn",
    return_value=((3, 3), 0.5, 0.1),
)
def test_nn_falls_back_to_neural_network_temperature_when_general_missing(
    mock_pick: MagicMock, _load: MagicMock
) -> None:
    cfg = _nn_config_base()
    cfg["general"].pop("temperature")
    analyze_position(
        _empty_board(),
        _test_black,
        "nn",
        cfg,
    )
    assert mock_pick.call_args.kwargs["temperature"] == 0.0
