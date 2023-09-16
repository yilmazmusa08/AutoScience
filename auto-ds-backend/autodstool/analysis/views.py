from django.http import HttpResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .serializers import AnalysisSerializer
from autodstool.utils.encoding import (determine_csv_encoding, determine_excel_df)
from autodstool.utils.models.init import *

class AnalysisViews(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = AnalysisSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            file = serializer.validated_data.get('file')
            target_column = serializer.validated_data.get('target_column')
            file_content = file.read()

            # Try to determine the encoding for CSV files
            df = determine_csv_encoding(file_content)

            if df is None:
                # If it's not a CSV, try reading it as an Excel file
                df = determine_excel_df(file_content)

            if df is None:
                return Response("Could not determine the correct encoding for the file.", status=status.HTTP_400_BAD_REQUEST)

            output = analysis(df=df, target=target_column)
            result_dict = calculate_pca(df.select_dtypes(include=['float', 'int']))
            output['PCA'] = {
                'Cumulative Explained Variance Ratio': result_dict['Cumulative Explained Variance Ratio'],
                'Principal Component': result_dict['Principal Component']
            }
            output = set_to_list(output)

            return Response({"Results": output})

        except Exception as e:
            return Response(f"Error occurred while processing the file: {str(e)}", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
