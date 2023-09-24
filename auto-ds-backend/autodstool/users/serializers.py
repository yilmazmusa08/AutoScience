from django.contrib.auth import get_user_model
from rest_framework import serializers
User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    date_joined = serializers.SerializerMethodField(read_only=True)
    last_login = serializers.SerializerMethodField(read_only=True)
    
    def get_date_joined(self, instance):
        return instance.date_joined.strftime("%Y-%m-%d %H:%M:%S")

    def get_last_login(self, instance):
        return instance.last_login.strftime("%Y-%m-%d %H:%M:%S")
    
    class Meta:
        model = User
        fields = (
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'date_joined',
            'last_login',
        )
