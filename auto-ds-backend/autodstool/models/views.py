from django.http import HttpResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .serializers import ModelsSerializer
from autodstool.utils.encoding import (determine_csv_encoding, determine_excel_df)
from autodstool.utils.models.init import *

class ModelsViews(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = ModelsSerializer(data=request.data)
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

            df = preprocess(df)
            problem_type, params = get_problem_type(df, target=target_column) # problem_type and params 
            output = create_model(df=df, problem_type=problem_type, params=params)
            # Convert NumPy array to list
            output = output.tolist() if isinstance(output, np.ndarray) else output

            return Response({"Results": output})

        except Exception as e:
            return Response(f"Error occurred while processing the file: {str(e)}", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
