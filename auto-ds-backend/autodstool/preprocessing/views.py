from django.http import HttpResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .serializers import PreprocessingSerializer
from autodstool.utils.models.init import *

class PreprocessingViews(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = PreprocessingSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            df = serializer.validated_data.get('file')
            # Perform your preprocessing steps here and save the preprocessed DataFrame to a CSV file
            # For example:
            preprocessed_df = preprocess(df)  # Replace with your preprocessing logic

            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="preprocessed.csv"'

            preprocessed_df.to_csv(response, index=False)
            return response

        except Exception as e:
            return Response(f"Error occurred while processing the file: {str(e)}", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
