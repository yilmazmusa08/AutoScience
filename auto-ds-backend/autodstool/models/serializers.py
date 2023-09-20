from django.conf import settings
from rest_framework import serializers
from autodstool.utils.encoding import (determine_csv_encoding, determine_excel_df)

# Define a custom validator function to check the file size
def validate_file_size(value):
    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if value.size > max_size_bytes:
        raise serializers.ValidationError(f"File size exceeds the maximum allowed size of {settings.MAX_FILE_SIZE_MB} MB.")

class ModelsSerializer(serializers.Serializer):
    file = serializers.FileField(use_url=False, validators=[validate_file_size])
    target_column = serializers.CharField(max_length=256, required=False)

    def validate_file(self, value):
        try:
            file_content = value.read()
            # Try to determine the encoding for CSV files
            df = determine_csv_encoding(file_content)
            if df is None:
                # If it's not a CSV, try reading it as an Excel file
                df = determine_excel_df(file_content)

            if df is None:
                raise serializers.ValidationError("Could not determine the correct encoding for the file.")
            return df

        except Exception as e:
            raise serializers.ValidationError("Could not determine the correct encoding for the file.")
