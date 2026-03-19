# fmt: off

from typing import Any

from common.exceptions import BadRequestError
from game_api.serializers import (AnalyzeRequestSerializer,
                                  AnalyzeResponseSerializer)
from game_api.services import analyze
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView, Request

# fmt: on


class HealthView(APIView):  # type: ignore
    authentication_classes: list[Any] = []
    permission_classes: list[Any] = []

    def get(self, request: Request) -> Response:
        return Response(
            {
                "status": "ok",
                "service": "lucidtree-api",
            },
            status=status.HTTP_200_OK,
        )


class AnalyzeView(APIView):  # type: ignore
    authentication_classes: list[Any] = []
    permission_classes: list[Any] = []

    def post(self, request: Request) -> Response:
        serializer = AnalyzeRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            result = analyze(serializer.validated_data)
        except BadRequestError as e:
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response(
                {"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        response_serializer = AnalyzeResponseSerializer(data=result)
        response_serializer.is_valid(raise_exception=True)

        return Response(response_serializer.data, status=status.HTTP_200_OK)
