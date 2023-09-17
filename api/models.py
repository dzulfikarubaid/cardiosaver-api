from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=50, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=8)
    c_password = models.CharField(max_length=8)
    # username = None

    # USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    def __str__(self):
        return self.name


class Answer(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    q1 = models.TextField(default='', blank=True)
    q2 = models.TextField(default='', blank=True)
    q3 = models.TextField(default='', blank=True)
    q4 = models.TextField(default='', blank=True)
    q5 = models.TextField(default='', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.name