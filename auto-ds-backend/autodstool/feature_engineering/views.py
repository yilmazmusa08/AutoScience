from django.http import HttpResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .serializers import FeatureEngineeringSerializer
from autodstool.utils.models.feature_engineering import perform_operation, take_first_n, take_last_n, create_transformed_column, replace_values, scale_column_in_dataframe, create_flag_column, remove_outliers


class FeatureEngineeringViews(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FeatureEngineeringSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            df = serializer.validated_data.get('file')
            operation = serializer.validated_data.get('operation')

            if operation == 'perform_operation':
                column_1 = serializer.validated_data.get('column_1')
                column_2 = serializer.validated_data.get('column_2')
                method = serializer.validated_data.get('method')
                df = perform_operation(df=df, column_1=column_1, column_2=column_2, method=method)

            elif operation == 'take_first_n':
                column_name = serializer.validated_data.get('column_name')
                n = serializer.validated_data.get('n')
                df = take_first_n(df=df, column_name=column_name, n=n)

            elif operation == 'take_last_n':
                column_name = serializer.validated_data.get('column_name')
                n = serializer.validated_data.get('n')
                df = take_last_n(df=df, column_name=column_name, n=n)

            elif operation == 'create_transformed_column':
                column_name = serializer.validated_data.get('column_name')
                transform_type = serializer.validated_data.get('transform_type')
                n = serializer.validated_data.get('n')
                df = create_transformed_column(df=df, column_name=column_name, transform_type=transform_type, n=n)

            elif operation == 'scale_column_in_dataframe':
                column_name = serializer.validated_data.get('column_name')
                scaler_type = serializer.validated_data.get('scaler_type')
                df = scale_column_in_dataframe(df=df, column_name=column_name, scaler_type=scaler_type)

            elif operation == 'replace_values':
                column_name = serializer.validated_data.get('column_name')
                value_to_search = serializer.validated_data.get('value_to_search')
                value_to_replace = serializer.validated_data.get('value_to_replace')
                df = replace_values(df=df, column_name=column_name, value_to_search=value_to_search, value_to_replace=value_to_replace)

            elif operation == 'create_flag_column':
                column_name = serializer.validated_data.get('column_name')
                value_to_search = serializer.validated_data.get('value_to_search')
                df = create_flag_column(df=df, column_name=column_name, value_to_search=value_to_search)
            
            elif operation == 'remove_outliers':
                column_name = serializer.validated_data.get('column_name')
                Quartile_1 = serializer.validated_data.get('Quartile_1')
                Quartile_3 = serializer.validated_data.get('Quartile_3')
                remove = serializer.validated_data.get('remove')
                df = remove_outliers(df=df, column_name=column_name, Quartile_1=Quartile_1, Quartile_3=Quartile_3, remove=remove)

                
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="engineered.csv"'

            df.to_csv(response, index=False)
            return response

        except Exception as e:
            return Response(f"Error occurred while processing the data: {str(e)}", status=status.HTTP_500_INTERNAL_SERVER_ERROR)
