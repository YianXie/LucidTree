from typing import Any

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView, Request


class HealthView(APIView):  # type: ignore
    authentication_classes: list[Any] = []
    permission_classes: list[Any] = []

    def get(self, request: Request) -> Response:
        return Response(
            {
                "status": "ok",
                "service": "mini-katago-api",
            },
            status=status.HTTP_200_OK,
        )
