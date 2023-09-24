from rest_framework.viewsets import GenericViewSet
from rest_framework.mixins import CreateModelMixin, ListModelMixin
from rest_framework.permissions import IsAuthenticated
from .models import Feedback
from .serializers import FeedbackSerializer

class FeedbackViews(ListModelMixin, CreateModelMixin, GenericViewSet):
    serializer_class = FeedbackSerializer
    queryset = Feedback.objects.all()
    permission_classes = [IsAuthenticated] 

    def perform_create(self, serializer):
        # Set the user for the new Feedback instance
        serializer.save(user=self.request.user)
