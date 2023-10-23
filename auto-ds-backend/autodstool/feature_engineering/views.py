from django.http import HttpResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .serializers import FeatureEngineeringSerializer
from autodstool.utils.models.feature_engineering import perform_operation, take_first_n, take_last_n, create_transformed_column, replace_values, create_flag_column, remove_outliers


class FeatureEngineeringViews(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FeatureEngineeringSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            df = serializer.validated_data.get('file')
            operation = serializer.validated_data.get('operation')

            if operation == 'perform_operation':
                col_1 = serializer.validated_data.get('col_1')
                col_2 = serializer.validated_data.get('col_2')
                method = serializer.validated_data.get('method')
                df = perform_operation(df=df, col_1=col_1, col_2=col_2, operation=method)

            elif operation == 'take_first_n':
                col_name = serializer.validated_data.get('col_name')
                n = serializer.validated_data.get('n')
                df = take_first_n(df=df, col_name=col_name, n=n)

            elif operation == 'take_last_n':
                col_name = serializer.validated_data.get('col_name')
                n = serializer.validated_data.get('n')
                df = take_last_n(df=df, col_name=col_name, n=n)

            elif operation == 'create_transformed_column':
                col_name = serializer.validated_data.get('col_name')
                transform_type = serializer.validated_data.get('transform_type')
                n = serializer.validated_data.get('n')
                df = create_transformed_column(df=df, col_name=col_name, transform_type=transform_type, n=n)

            elif operation == 'scale_column_in_dataframe':
                col_name = serializer.validated_data.get('col_name')
                scaler_type = serializer.validated_data.get('scaler_type')
                df = create_transformed_column(df=df, col_name=col_name, scaler_type=scaler_type, n=n)

            elif operation == 'replace_values':
                col_name = serializer.validated_data.get('col_name')
                val_search = serializer.validated_data.get('val_search')
                val_replace = serializer.validated_data.get('val_replace')
                df = replace_values(df=df, col_name=col_name, val_search=val_search, val_replace=val_replace)

            elif operation == 'create_flag_column':
                col_name = serializer.validated_data.get('col_name')
                val_search = serializer.validated_data.get('val_search')
                df = create_flag_column(df=df, col_name=col_name, val_search=val_search)
            
            elif operation == 'remove_outliers':
                col_name = serializer.validated_data.get('col_name')
                Q1 = serializer.validated_data.get('Q1')
                Q3 = serializer.validated_data.get('Q3')
                remove = serializer.validated_data.get('remove')
                df = remove_outliers(df=df, col_name=col_name, Q1=Q1, Q3=Q3, remove=remove)

                
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="engineered.csv"'

            df.to_csv(response, index=False)
            return response

        except Exception as e:
            return Response(f"Error occurred while processing the data: {str(e)}", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
