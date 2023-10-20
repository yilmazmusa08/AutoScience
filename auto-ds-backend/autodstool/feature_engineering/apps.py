from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _

class FeatureEngineeringConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'autodstool.feature_engineering'
    verbose_name = _("Feature Engineering")