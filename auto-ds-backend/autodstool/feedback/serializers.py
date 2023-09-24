from rest_framework import serializers
from .models import Feedback
from autodstool.users.serializers import UserSerializer

class FeedbackSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = Feedback
        fields = '__all__'