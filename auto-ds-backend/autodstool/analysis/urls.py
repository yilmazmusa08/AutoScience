from django.urls import path
from .views import (
    AnalysisViews,
)

urlpatterns = [
    path("analyze/", AnalysisViews.as_view(), name="analyze"),
]
