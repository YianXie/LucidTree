from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.admin.sites import settings
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("game_api.urls")),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
