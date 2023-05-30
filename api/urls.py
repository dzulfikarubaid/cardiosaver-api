from . import views
from django.urls import path, re_path
from django.views.generic import TemplateView

urlpatterns = [
    path('api/register/', views.Register),
    path('api/login/', views.Login),
    path('api/user/', views.UserView),
    path('api/addpost/', views.AddPost),
    path('api/post/', views.ViewPost),
    path('api/profile/<int:id>/', views.Profile),
    path('api/apply/', views.Apply),
    path('api/apply-list/<int:id>/', views.ApplyList),
    re_path(r'.*', views.index)
]