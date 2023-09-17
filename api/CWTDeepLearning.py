import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
import matplotlib.pyplot as plt
import random

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
        # print(f'skala[{i}] = {fk} Hz, waktu[{j}] = {tk} s')
    return cwt

def morlet_complex(t, a, b, w0):
    real = 1 / np.sqrt(a) * 1 / np.sqrt(np.pi) * np.exp(-((t - b) ** 2) / (2 * a ** 2)) * np.cos(w0 * (t - b) / a)
    imag = 1 / np.sqrt(a) * (-1) * 1 / np.sqrt(np.pi) * np.exp(-((t - b) ** 2) / (2 * a ** 2)) * np.sin(w0 * (t - b) / a)
    return real, imag

# Contoh penggunaan:
def process_data_file(data, parameter):
    signal = data

    if signal is not None:
        a0 = 0.0001
        b0 = 0
        da = 0.0009

        cwt_result = cwt(signal, morlet_complex, a0, b0, da)

        # Simpan hasil CWT sebagai gambar
        output_image_path = f"./result_{parameter}.png"
        plt.imshow(cwt_result, aspect='auto', extent=[0, 10, 1, 100])
        plt.title('Continuous Wavelet Transform (CWT)')
        plt.xlabel('Waktu (detik)')
        plt.ylabel('Skala')
        plt.colorbar(label='Magnitude')
        plt.savefig(output_image_path)
        plt.close()  # Ganti plt.clf() dengan plt.close() untuk menutup gambar

        # Ganti dengan path gambar CWT yang ingin Anda gunakan sebagai input
        cwt_image_path = output_image_path

        # Load the model
        model_path = './model_dl.h5'  # Ganti dengan path ke model deep learning Anda
        model = load_model(model_path)

        # Load and preprocess the CWT image
        img = tf.keras.preprocessing.image.load_img(cwt_image_path, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        normalized_img_array = img_array / 255.0  # Normalisasi ke [0, 1]
        expanded_img_array = np.expand_dims(normalized_img_array, axis=0)

        # Perform prediction
        predictions = model.predict(expanded_img_array)

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

        # Classify the prediction
        risk_category = classify_risk(predictions[0])

        # Print the risk category
        return predictions[0], risk_category

# Panggil fungsi untuk memproses data file
process_data_file(4)  # Ganti dengan indeks file yang sesuai