from . import views
from django.urls import path, re_path
from django.views.generic import TemplateView

urlpatterns = [
    path('api/register/', views.Register),
    path('api/login/', views.Login),
    path('api/user/', views.UserView),
    path('api/addpost/', views.AddPost),
    path('api/post/', views.ViewPost),
    re_path(r'.*', views.index)
]