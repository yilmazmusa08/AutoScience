from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _

class PreprocessingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'autodstool.preprocessing'
    verbose_name = _("Preprocessing")
