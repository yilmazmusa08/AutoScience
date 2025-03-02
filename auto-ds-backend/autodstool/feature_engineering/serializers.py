from django.conf import settings
from rest_framework import serializers
from autodstool.utils.encoding import (determine_csv_encoding, determine_excel_df)

# Define a custom validator function to check the file size
def validate_file_size(value):
    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if value.size > max_size_bytes:
        raise serializers.ValidationError(f"File size exceeds the maximum allowed size of {settings.MAX_FILE_SIZE_MB} MB.")

OPERATION_CHOICES = (
    ('perform_operation', 'Perform Operation'),
    ('take_first_n', 'Take First N'),
    ('take_last_n', 'Take Last N'),
    ('create_transformed_column', 'Create Transformed Column'),
    ('scale_column_in_dataframe', 'Normalize The Column'),
    ('replace_values', 'Replace Values'),
    ('create_flag_column', 'Create Flag Column'),
    ('remove_outliers', 'Remove Outliers'),
)

METHOD_CHOICES = (
    ('add', 'Addition Operation'),
    ('substract', 'Substraction Operation'),
    ('multiply', 'Multiplication Operation'),
    ('divide', 'Division Operation'),
)

TRANSFORM_TYPE_CHOICES = (
    ('log', 'Logarithm Operation'),
    ('power', 'Power Operation'),
    ('root', 'Root Operation'),
)

SCALER_TYPE_CHOICES = (
    ('StandardScaler', 'StandardScaler Operation'),
    ('MinMaxScaler', 'MinMaxScaler Operation'),
    ('RobustScaler', 'RobustScaler Operation'),
)


class FeatureEngineeringSerializer(serializers.Serializer):
    file = serializers.FileField(use_url=False, validators=[validate_file_size])
    operation = serializers.ChoiceField(choices=OPERATION_CHOICES)
    column_name = serializers.CharField(max_length=256, required=False)
    column_1 = serializers.CharField(max_length=256, required=False)
    column_2 = serializers.CharField(max_length=256, required=False)
    method = serializers.ChoiceField(choices=METHOD_CHOICES, required=False)
    n = serializers.IntegerField(required=False)
    Quartile_1 = serializers.IntegerField(required=False, min_value=0, max_value=100)
    Quartile_3 = serializers.IntegerField(required=False, min_value=0, max_value=100)
    remove = serializers.BooleanField(required=False)
    transform_type = serializers.ChoiceField(choices=TRANSFORM_TYPE_CHOICES, required=False)
    scaler_type = serializers.ChoiceField(choices=SCALER_TYPE_CHOICES, required=False)
    value_to_search = serializers.CharField(max_length=256, required=False)
    value_to_replace = serializers.CharField(max_length=256, required=False)

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
