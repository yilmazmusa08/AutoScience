from django.http import HttpResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .serializers import ModelsSerializer
from autodstool.utils.models.init import *

class ModelsViews(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = ModelsSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            df = serializer.validated_data.get('file')
            target_column = serializer.validated_data.get('target_column')

            df = preprocess(df)
            problem_type, params = get_problem_type(df, target=target_column) # problem_type and params 
            output = create_model(df=df, problem_type=problem_type, params=params)
            # Convert NumPy array to list
            output = output.tolist() if isinstance(output, np.ndarray) else output

            return Response({"Results": output})

        except Exception as e:
            return Response(f"Error occurred while processing the models: {str(e)}", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
