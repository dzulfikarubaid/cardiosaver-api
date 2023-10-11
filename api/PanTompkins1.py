import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Membaca sinyal EKG dari file (ganti 'nama_file.txt' dengan nama file yang sesuai)
# File harus berisi data EKG dalam satu kolom dengan nilai per sampel
# ekg_signal = np.loadtxt("C:/Users/ASUS/OneDrive/Documents/Kuliahh/Lomba/PKM/PKM 2023/PKM KC/Data/Data Arythmia SCD/30/data_bagian_1.csv", skiprows=1)  # Baca sinyal dari file
data =[1821,2821,21812,1213,2322,2332,3233,1212,2121,2121,2121,2121,1221,2121,2333]
def get_bpm_rr(data):
    ekg_signal= np.array(data)
# Sampling frequency (gantilah sesuai dengan frekuensi sampling data Anda)
    fs = 100 # Contoh: 1000 Hz

    # Time vector
    t = np.arange(0, len(ekg_signal)) / fs

    # Find R-peaks using a peak detection algorithm (you can adjust parameters)
    peaks, _ = find_peaks(ekg_signal, height=0.05, distance=0.6*fs)

    # Calculate heart rate based on intervals between R-peaks
    tt = 1.0 / fs  # Faktor konversi waktu
    interval = np.diff(peaks) * tt  # Menghitung interval antara puncak R
    bpm = 60.0 / interval  # Menghitung denyut jantung berdasarkan interval

    # Plot the EKG signal with detected R-peaks
    plt.figure(figsize=(12, 6))
    plt.plot(t, ekg_signal, label='EKG Signal', color='b')
    plt.scatter(np.array(peaks)/fs, ekg_signal[peaks], color='r', marker='o', label='Detected R-Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('EKG Signal with R-Peak Detection')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Heart Rate: {np.mean(bpm):.2f} bpm")  # Menggunakan rata-rata denyut jantung dari interval
    print(f"Interval RR: {np.mean(interval):.2f} sekon")

get_bpm_rr(data)