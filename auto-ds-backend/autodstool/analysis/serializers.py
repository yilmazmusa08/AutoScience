from rest_framework import serializers
from autodstool.utils.encoding import (determine_csv_encoding, determine_excel_df)

class AnalysisSerializer(serializers.Serializer):
    file = serializers.FileField(use_url=False)
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