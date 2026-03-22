# fmt: off

import logging
from typing import Any

from common.exceptions import BadRequestError
from game_api.models import AnalyzeRequest
from game_api.serializers import (AnalyzeRequestSerializer,
                                  AnalyzeResponseSerializer)
from game_api.services import analyze
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView, Request

# fmt: on

logger = logging.getLogger(__name__)


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
        request_serializer = AnalyzeRequestSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)

        request_model_instance = AnalyzeRequest(data=request_serializer.validated_data)
        request_model_instance.save()

        try:
            result = analyze(request_serializer.validated_data)
        except BadRequestError as e:
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            logger.exception("Unexpected error during position analysis")
            return Response(
                {"detail": "An internal server error occurred."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        response_serializer = AnalyzeResponseSerializer(data=result)
        response_serializer.is_valid(raise_exception=True)

        return Response(response_serializer.data, status=status.HTTP_200_OK)
