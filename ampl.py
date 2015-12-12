# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:49:40 2015

@author: User
"""

# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal
import glob
import wave
from scipy.signal import butter, filtfilt

wavs_train=[]   # амплитуды файлов обучающей выборки
wavs_test = []   # амплитуды файлов тестовой выборки

def pre_proc(w):   # предобработка звука
    frame_rate = w.getframerate()    
    samples = np.frombuffer(w.readframes(w.getnframes()), np.dtype(np.int16))
    # стандартизация сигнала
    samples = (4000 * (samples - np.mean(samples)) / np.std(samples)).astype('int16')
    # фильтрация сигналом ФНЧ с полосой 4 кГц
    b, a = butter(16, 4000 / (frame_rate / 2))
    samples = filtfilt(b, a, samples).astype('int16')
    # окно и порог для удаления пауз в речевом сигнале
    weights_size = 1000
    weights = np.repeat([1.0], weights_size) / weights_size
    silence_threshold = 600
    # удаление паузы
    smooth = np.convolve(np.abs(samples), weights, 'same')
    samples = samples[smooth > silence_threshold]   
    return samples


for filename in glob.glob("D:\\mmad\\kurs\\train\\*.wav"):  
    w = wave.open(filename, 'rb')  # чтение файло обучающей выборки
    wavs_train.append(pre_proc(w)) # записываем в массив амплитуды предварительно обработанных файлов

for filename in glob.glob("D:\\mmad\\kurs\\test\\*.wav"): 
    w = wave.open(filename, 'rb')  # чтение файлов тестовой выборки
    wavs_test.append(pre_proc(w))  # записываем в массив амплитуды предварительно обработанных файлов

#преобразуем list в массив numpy        
train_ampl = np.array(wavs_train)  
test_ampl = np.array(wavs_test)  

#сохраняем амплитуды в бинарные файлы
np.save("train_ampl.npy", train_ampl)
np.save("test_ampl.npy",test_ampl)


