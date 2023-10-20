from django.urls import path
from .views import (
    FeatureEngineeringViews,
)

urlpatterns = [
    path("feature_engineering/", FeatureEngineeringViews.as_view(), name="feature_engineering"),
]
