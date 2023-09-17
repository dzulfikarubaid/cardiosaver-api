from django.contrib import admin
from . import models
from .models import Answer, User
# Register your models here.
admin.site.register([User, Answer])