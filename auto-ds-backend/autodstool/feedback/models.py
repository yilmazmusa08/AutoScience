from django.db import models
from django.contrib.auth import get_user_model
User=get_user_model()

class Feedback(models.Model):
    feedback = models.CharField(max_length=500)
    user = models.ForeignKey(User, related_name='users', on_delete=models.PROTECT)
    created_at = models.DateTimeField(auto_now_add=True)
