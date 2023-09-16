from rest_framework import serializers

class AnalysisSerializer(serializers.Serializer):
    file = serializers.FileField(use_url=False)
    target_column = serializers.CharField(max_length=256, required=False)