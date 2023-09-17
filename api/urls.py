from . import views
from django.urls import path, re_path
from django.views.generic import TemplateView

urlpatterns = [
    path('api/data/', views.data),
    path('api/data_dl/', views.data_result),
    path('api/register/', views.Register),
    path('api/login/', views.Login),
    path('api/answer/', views.create_answer),
]