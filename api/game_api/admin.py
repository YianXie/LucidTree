from django.contrib import admin

from .models import AnalyzeRequest


class AnalyzeRequestAdmin(admin.ModelAdmin):  # type: ignore
    """
    Admin for the analyze request model

    Returns:
        AnalyzeRequestAdmin: The admin instance
    """

    list_display = ("id", "created_at")
    readonly_fields = ("id", "created_at")


admin.site.register(AnalyzeRequest, AnalyzeRequestAdmin)
