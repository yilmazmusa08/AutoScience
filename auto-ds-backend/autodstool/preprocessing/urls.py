from django.urls import path
from autodstool.preprocessing.views import (
    PreprocessingViews,
)

urlpatterns = [
    path("preprocessing/", PreprocessingViews.as_view(), name="preprocessing"),
]
