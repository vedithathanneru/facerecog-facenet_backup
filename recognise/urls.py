from django.urls import path
from .views import recognize_from_form, check_embedding_status 

urlpatterns = [
    path('recognise/', recognize_from_form),
    path('check/', check_embedding_status)
]