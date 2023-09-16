from django.urls import path
from .views import (
    PreprocessingViews,
)

urlpatterns = [
    path("preprocess/", PreprocessingViews.as_view(), name="preprocess"),
]
