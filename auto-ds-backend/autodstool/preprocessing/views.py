import io
import pandas as pd
from django.http import HttpResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .serializers import FileUploadSerializer

class PreprocessingViews(APIView):
    permission_classes = [IsAuthenticated]

    def determine_csv_encoding(self, file_content):
        csv_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'utf-32']
        for encoding in csv_encodings:
            try:
                return pd.read_csv(io.BytesIO(file_content), encoding=encoding)
            except (pd.errors.ParserError, UnicodeDecodeError):
                continue
        return None

    def determine_excel_df(self, file_content):
        try:
            return pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
        except (pd.errors.ParserError, UnicodeDecodeError):
            return None

    def post(self, request):
        serializer = FileUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            file = serializer.validated_data['file']
            file_content = file.read()

            # Try to determine the encoding for CSV files
            df = self.determine_csv_encoding(file_content)

            if df is None:
                # If it's not a CSV, try reading it as an Excel file
                df = self.determine_excel_df(file_content)

            if df is None:
                return Response("Could not determine the correct encoding for the file.", status=status.HTTP_400_BAD_REQUEST)

            # Perform your preprocessing steps here and save the preprocessed DataFrame to a CSV file
            # For example:
            preprocessed_df = df  # Replace with your preprocessing logic

            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="preprocessed.csv"'

            preprocessed_df.to_csv(response, index=False)
            return response

        except Exception as e:
            return Response(f"Error occurred while processing the file: {str(e)}", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
