from django.conf import settings
from rest_framework.routers import DefaultRouter, SimpleRouter

from autodstool.feedback.views import (
    FeedbackViews,
)

if settings.DEBUG:
    router = DefaultRouter()
else:
    router = SimpleRouter()

router.register("feedback", FeedbackViews)

app_name = "api"
urlpatterns = router.urls
