import uuid

from django.db import models


class AnalyzeRequest(models.Model):  # type: ignore
    """
    Analyze request model

    Returns:
        str: The string representation of the analyze request
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"AnalyzeRequest {self.id}"
