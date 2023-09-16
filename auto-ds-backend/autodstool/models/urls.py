from django.urls import path
from .views import (
    ModelsViews,
)

urlpatterns = [
    path("models/", ModelsViews.as_view(), name="models"),
]
