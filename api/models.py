from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=50, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField()
    c_password = models.CharField()
    username = None

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    def __str__(self):
        return self.name

class Post(models.Model):
    id_post = models.AutoField(primary_key=True, unique=True)
    nama_item = models.CharField(max_length=50)
    perkiraan_harga_item = models.IntegerField()
    biaya_titip = models.IntegerField()
    alamat_nitiper = models.CharField(max_length=250)
    alamat_pembelian = models.CharField(max_length=250)
    nitiper = models.ForeignKey(User, on_delete=models.CASCADE)
    gambar = models.CharField()

    def __str__(self):
        return self.nitiper + ' ' + self.nama_item

class Interaction(models.Model):
    id_interaction = models.AutoField(primary_key=True, unique=True)
    nitiper = models.ForeignKey(User, on_delete=models.CASCADE)
    respons = models.ForeignKey(Post, on_delete=models.CASCADE)
    sender = models.CharField()