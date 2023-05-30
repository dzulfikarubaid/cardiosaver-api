from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.exceptions import AuthenticationFailed
from .serializers import UserSerializer, PostSerializer, UpdateUserSerializer, InteractionSerializer
from .models import User, Post, Interaction
import jwt, datetime
import requests
# Create your views here.

def index(request):
    return render(request, 'base.html')


@api_view(['GET'])
def ApplyList(request,id):
    try:
        apply = list(Interaction.objects.filter(nitiper=id))
    except Interaction.DoesNotExist:
        status = {'status':'data tidak ditemukan'}
        return Response(status)
    serializer = InteractionSerializer(apply, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def Apply(request):
    serializer = InteractionSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors)
@api_view(['PUT'])
def Profile(request, id):
    name = request.data['name']
    user = User.objects.filter(name=name).first()
    try:
        ModelObject = User.objects.get(id=id)
    except:
        status = {'status':'data tidak ditemukan'}
        return Response(status)
    serializer = UpdateUserSerializer(ModelObject, data=request.data)
    
    if serializer.is_valid():
        serializer.save()
        payload = {
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'password': user.password,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=60),
        'iat': datetime.datetime.utcnow()
        }
        token = jwt.encode(payload, 'secret', algorithm='HS256')
        response = Response()
        response.data = {
            'jwt': token
        }
        return response
    return Response(serializer.errors)
@api_view(['GET'])
def ViewPost(request):
    posts = Post.objects.all()
    serializer = PostSerializer(posts, many=True)
    nitiper_ids = []
    nitiper_names = []
    for post in posts:
        nitiper_ids.append(post.nitiper_id)
    for nitiper_id in nitiper_ids:
        nitiper_name = User.objects.get(id=nitiper_id).name
        nitiper_names.append(nitiper_name)
    response_data = []
    for post, nitiper_id, nitiper_name in zip(posts, nitiper_ids, nitiper_names):
        response_data.append({
            "id_post": post.id_post,
            "nama_item": post.nama_item,
            "perkiraan_harga_item": post.perkiraan_harga_item,
            "biaya_titip": post.biaya_titip,
            "alamat_nitiper": post.alamat_nitiper,
            "alamat_pembelian": post.alamat_pembelian,
            "nitiper": nitiper_id,
            "nama": nitiper_name
        })
    return Response(response_data)

@api_view(['POST'])
def AddPost(request):
    # token = request.COOKIES.get('jwt')
    # if not token:
    #     raise AuthenticationFailed('Unauthenticated!')
    nitiper = request.data['nitiper']
    nama = User.objects.filter(id=nitiper).values()
    serializer = PostSerializer(data=request.data)
    if serializer.is_valid(raise_exception=True):
        serializer.save()
    return Response(serializer.data)
@api_view(['POST'])
def Register(request):
    serializer = UserSerializer(data=request.data)
    password = request.data['password']
    c_password = request.data['c_password']
    if password != c_password:
        raise AuthenticationFailed('Passwords do not match!')
    if serializer.is_valid(raise_exception=True):
        serializer.save()
    return Response(serializer.data)

@api_view(['POST'])
def Login(request):
    email = request.data['email']
    password = request.data['password']
    user = User.objects.filter(email=email).first()

    if email == '' or password == '':
        raise AuthenticationFailed('Please provide email and password!')
    if user is None:
        raise AuthenticationFailed('User not found!')
    if not user.check_password(password):
        raise AuthenticationFailed('Incorrect password!')
    payload = {
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'password': user.password,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=60),
        'iat': datetime.datetime.utcnow()
    }
    token = jwt.encode(payload, 'secret', algorithm='HS256')
    response = Response()
    response.set_cookie(key='jwt', value=token, httponly=True)
    response.data = {
        'jwt': token
    }
    return response

@api_view(['GET'])
def UserView(request):
    token = request.COOKIES.get('jwt')
    if not token:
        raise AuthenticationFailed('Unauthenticated!')
    try:
        payload = jwt.decode(token, 'secret', algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        raise AuthenticationFailed('Unauthenticated!')
    
    user = User.objects.filter(id=payload['id']).first()
    serializer = UserSerializer(user)
    return Response(serializer.data)