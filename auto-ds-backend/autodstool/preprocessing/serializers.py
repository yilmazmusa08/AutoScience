from rest_framework import serializers

class PreprocessingSerializer(serializers.Serializer):
    file = serializers.FileField(use_url=False)