from django.http import HttpResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .serializers import AnalysisSerializer
from autodstool.utils.models.init import *

class AnalysisViews(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = AnalysisSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            df = serializer.validated_data.get('file')
            target_column = serializer.validated_data.get('target_column')

            df = clean_dataframe(df)
            df = preprocess(df)
            output = analysis(df=df, target=target_column)
            result_dict = calculate_pca(df.select_dtypes(include=['float', 'int']))
            output['PCA'] = {
                'Cumulative Explained Variance Ratio': result_dict['Cumulative Explained Variance Ratio'],
                'Principal Component': result_dict['Principal Component']
            }
            output = set_to_list(output)

            return Response({"Results": output})

        except Exception as e:
            return Response(f"Error occurred during analysis: {str(e)}", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
