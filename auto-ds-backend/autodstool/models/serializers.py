from rest_framework import serializers

class ModelsSerializer(serializers.Serializer):
    file = serializers.FileField(use_url=False)
    target_column = serializers.CharField(max_length=256, required=False)