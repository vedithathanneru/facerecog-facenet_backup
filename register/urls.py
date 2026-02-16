from django.urls import path
from .views import GenerateUserEmbeddingsViewForm

urlpatterns = [
    path('form/', GenerateUserEmbeddingsViewForm.as_view()),
]