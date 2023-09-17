from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.exceptions import AuthenticationFailed
from .serializers import UserSerializer, AnswerSerializer
from .models import User, Answer
import jwt, datetime
import requests
from rest_framework import status
import pyrebase
from scipy.signal import butter, filtfilt, iirnotch
from decouple import config
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
import matplotlib.pyplot as plt
import os

config={
    "apiKey": config("FIREBASE_API_KEY2"),
    "authDomain": config("FIREBASE_AUTH_DOMAIN2"),
    "databaseURL": config("FIREBASE_DATABASE_URL2"),
    "projectId": config("FIREBASE_PROJECT_ID2"),
    "storageBucket": config("FIREBASE_STORAGE_BUCKET2"),
    "messagingSenderId": config("FIREBASE_MESSAGING_SENDER_ID2"),
    "appId": config("FIREBASE_APP_ID2")
}
firebase=pyrebase.initialize_app(config)
authe = firebase.auth()
database=firebase.database()
row = 100
column = 100

def read_data_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = []

            for line in lines[1:]:  # Mulai dari baris kedua
                value = line.strip()
                data.append(float(value))

            return data
    except Exception as e:
        print('Gagal membaca file:', e)
        return None

def cwt(signal, wavelet, a0, b0, da):
    ndata = len(signal)
    dt = 1 / 1000  # fs = 1000
    t = np.arange(0, ndata * dt, dt)

    f0 = 0.849  # frekuensi center
    w0 = 2 * math.pi * f0
    db = (ndata - 1) * dt / column
    cwt = np.zeros((row, column))

    a = a0
    for i in range(row):
        b = b0
        for j in range(column):
            real, imag = wavelet(t, a, b, w0)

            real_cwt = 0
            imag_cwt = 0

            for k in range(ndata):
                kernel_real, kernel_imag = real[k], imag[k]
                real_cwt += signal[k] * kernel_real
                imag_cwt += signal[k] * kernel_imag

            cwt[i][j] = np.sqrt(real_cwt ** 2 + imag_cwt ** 2)
            b += db
        a += da

    fc = f0
    for i in range(1, row + 1):
        fk = fc / (a0 + i * da)
        j = i
        tk = (ndata - 1) / column * j * dt
        print(f'skala[{i}] = {fk} Hz, waktu[{j}] = {tk} s')
    return cwt

def morlet_complex(t, a, b, w0):
    real = 1 / np.sqrt(a) * 1 / np.sqrt(np.pi) * np.exp(-((t - b) ** 2) / (2 * a ** 2)) * np.cos(w0 * (t - b) / a)
    imag = 1 / np.sqrt(a) * (-1) * 1 / np.sqrt(np.pi) * np.exp(-((t - b) ** 2) / (2 * a ** 2)) * np.sin(w0 * (t - b) / a)
    return real, imag

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Apply BPF to the signal
def apply_bandpass_filter(data, lowcut, highcut, fs=1000, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Define a function to apply a notch filter
def apply_notch_filter(data, fs=1000, Q=30.0, f0=50.0):
    # Calculate the notch frequency in radians per sample
    w0 = f0 / (0.5 * fs)
    
    # Design the notch filter
    b, a = iirnotch(w0, Q)
    
    # Apply the notch filter to the data
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

# Define risk categories
def classify_risk(prediction_value):
    if prediction_value <= 0.3:
        return "risiko rendah"
    elif 0.4 <= prediction_value <= 0.7:
        return "risiko sedang"
    elif prediction_value >= 0.8:
        return "risiko tinggi"
    else:
        return "Unknown"  # Handle unexpected values

# penggunaan:
def process_data_file(signal, parameter):
    
    if signal is not None:
        a0 = 0.0001
        b0 = 0
        da = 0.0009

        # Apply BPF to the signal
        lowcut = 0.05  
        highcut = 100.0  
        filtered_signal_bpf = apply_bandpass_filter(signal, lowcut, highcut)

        # Apply the notch filter
        notch_filtered_signal = apply_notch_filter(filtered_signal_bpf)

        cwt_result = cwt(notch_filtered_signal, morlet_complex, a0, b0, da)

        # Simpan hasil CWT sebagai gambar
        output_image_path = f"./result_{parameter}.png"
        plt.imshow(cwt_result, aspect='auto', extent=[0, 10, 1, 100])
        plt.title('Continuous Wavelet Transform (CWT)')
        plt.xlabel('Waktu (detik)')
        plt.ylabel('Skala')
        plt.colorbar(label='Magnitude')
        plt.savefig(output_image_path)
        plt.close()  

        # Ganti dengan path gambar CWT yang ingin Anda gunakan sebagai input
        cwt_image_path = output_image_path

        # Load the model
        model_path = './my_modelCWT1.h5'  
        model = load_model(model_path)

        # Load and preprocess the CWT image
        img = tf.keras.preprocessing.image.load_img(cwt_image_path, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        normalized_img_array = img_array / 255.0  # Normalisasi ke [0, 1]
        expanded_img_array = np.expand_dims(normalized_img_array, axis=0)

        # Perform prediction
        predictions = model.predict(expanded_img_array)

        # Classify the prediction
        risk_category = classify_risk(predictions[0])

        # Print the risk category
        return predictions[0], risk_category



def result(q1,q2,q3,q4,q5):
    if q1 == 'A':
        q1 = 20
    else:
        q1 = 0
    if q2 == 'A':
        q2 = 20
    else:
        q2 = 0
    if q3 == 'A':
        q3 = 20
    else:
        q3 = 0
    if q4 == 'A':
        q4 = 20
    else:
        q4 = 0
    if q5 == 'A':
        q5 = 20
    else:
        q5 = 0
    
    return q1+q2+q3+q4+q5
@api_view(['GET'])
def data(request):
    value = database.child('test').child('int').get().val().values()
    return Response(value)
@api_view(['GET'])
def data_result(request):
    result_tuple = process_data_file(list(database.child('test').child('int').get().val().values()), 'test')  # Panggil fungsi dan simpan hasilnya dalam tuple
    predictions = str(result_tuple[0])  # Ambil nilai pertama dari tuple (predictions)
    risk_cat = str(result_tuple[1])  # Ambil nilai kedua dari tuple (risk_cat)
    return Response({
        'predictions': predictions,  # Gunakan nama string sebagai kunci
        'risk_cat': risk_cat
    })

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

@api_view(['POST'])
def create_answer(request):
    serializer = AnswerSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        q1 = serializer.data['q1']
        q2 = serializer.data['q2']
        q3 = serializer.data['q3']
        q4 = serializer.data['q4']
        q5 = serializer.data['q5']
        answer_result = result(q1,q2,q3,q4,q5)
        
        return Response(answer_result, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)