from rest_framework import serializers

class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField(use_url=False)
    target_column = serializers.CharField(max_length=100, required=False)