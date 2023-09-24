from django.contrib import admin
from django.db import models
from django.forms import Textarea
from .models import Feedback

from import_export.admin import ImportExportModelAdmin
# Register your models here.

@admin.register(Feedback)
class FeedbackAdmin(ImportExportModelAdmin):
    list_display = ('id', 'user', 'feedback', 'created_at')
    search_fields = ('user__username', 'user__email', 'feedback')
    list_filter = ('user__username',)

    formfield_overrides = {
        models.CharField: {'widget': Textarea(attrs={'rows': 8, 'cols': 80})},
    }
