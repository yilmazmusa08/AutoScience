from django.utils.translation import gettext_lazy as _
from django.apps import AppConfig

class AnalysisConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'autodstool.analysis'
    verbose_name = _("Analysis")
    