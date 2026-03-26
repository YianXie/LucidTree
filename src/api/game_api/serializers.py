from typing import Any

from rest_framework import serializers

ALGO_CHOICES = ("minimax", "nn", "mcts")
PLAYER_CHOICES = ("B", "W")
BOARD_SIZE_CHOICES = (9, 13, 19)
RULE_CHOICES = ("japanese", "chinese")


class AnalyzeParamsSerializer(serializers.Serializer):  # type: ignore
    """
    Serializer for the analyze parameters

    Returns:
        dict[str, Any]: The validated parameters, will be used to pass to the analyze function
    """

    # MCTS params
    num_simulations = serializers.IntegerField(
        required=False, min_value=1, max_value=5000
    )
    c_puct = serializers.FloatField(required=False, min_value=0.0, max_value=10.0)

    # MiniMax params
    depth = serializers.IntegerField(required=False, min_value=1, max_value=6)

    # NN params
    model_name = serializers.CharField(required=False, max_length=100)

    # Shared optional params
    seed = serializers.IntegerField(required=False)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        return attrs


class AnalyzeRequestSerializer(serializers.Serializer):  # type: ignore
    """
    Serializer for the analyze requests

    Raises:
        serializers.ValidationError:
        serializers.ValidationError: If the moves are not valid
        serializers.ValidationError: If the color is not valid
        serializers.ValidationError: If the point is not valid

    Returns:
        list[tuple[str, str]]: The validated moves
    """

    board_size = serializers.ChoiceField(choices=BOARD_SIZE_CHOICES)
    rules = serializers.ChoiceField(choices=RULE_CHOICES, default="japanese")
    komi = serializers.FloatField(required=False, default=6.5)
    to_play = serializers.ChoiceField(choices=PLAYER_CHOICES)

    # Accept moves as list of [color, point]
    moves = serializers.ListField(
        child=serializers.ListField(
            child=serializers.CharField(),
            min_length=2,
            max_length=2,
        ),
        required=False,
        default=list,
    )

    algo = serializers.ChoiceField(choices=ALGO_CHOICES)
    params = AnalyzeParamsSerializer(required=False, default=dict)

    def validate_moves(self, value: list[list[str]]) -> list[tuple[str, str]]:
        validated: list[tuple[str, str]] = []

        for i, item in enumerate(value):
            if len(item) != 2:
                raise serializers.ValidationError(
                    f"Each move must be [color, point]. Error at index {i}."
                )

            color, point = item
            color = str(color).strip().upper()
            point = str(point).strip().upper()

            if color not in PLAYER_CHOICES:
                raise serializers.ValidationError(
                    f"Invalid color '{color}' at moves[{i}]. Must be 'B' or 'W'."
                )

            if not point:
                raise serializers.ValidationError(
                    f"Move point cannot be empty at moves[{i}]."
                )

            validated.append((color, point))

        return validated


class AnalyzeResponseSerializer(serializers.Serializer):  # type: ignore
    """
    Serializer for the analyze responses

    Returns:
        AnalyzeResponseSerializer: The serializer instance
    """

    best_move = serializers.CharField(required=True)
    algorithm = serializers.CharField(required=True)
    stats = serializers.DictField(required=True)
