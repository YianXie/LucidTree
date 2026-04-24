from django.urls import path

from api.game_api import views

urlpatterns = [
    path("health/", views.HealthView.as_view(), name="health"),
    path("analyze/", views.AnalyzeView.as_view(), name="analyze"),
    path("winrate/", views.WinrateView.as_view(), name="winrate"),
]
