from django.http import HttpResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .serializers import FileUploadSerializer
from autodstool.utils.encoding import (determine_csv_encoding, determine_excel_df)
from autodstool.utils.models.init import *

class PreprocessingViews(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FileUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            file = serializer.validated_data.get('file')
            file_content = file.read()

            # Try to determine the encoding for CSV files
            df = determine_csv_encoding(file_content)

            if df is None:
                # If it's not a CSV, try reading it as an Excel file
                df = determine_excel_df(file_content)

            if df is None:
                return Response("Could not determine the correct encoding for the file.", status=status.HTTP_400_BAD_REQUEST)

            # Perform your preprocessing steps here and save the preprocessed DataFrame to a CSV file
            # For example:
            preprocessed_df = preprocess(df)  # Replace with your preprocessing logic

            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="preprocessed.csv"'

            preprocessed_df.to_csv(response, index=False)
            return response

        except Exception as e:
            return Response(f"Error occurred while processing the file: {str(e)}", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
