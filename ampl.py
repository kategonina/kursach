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



wavs_train=[]
wavs_test = []

def pre_proc(w):
    frame_rate = w.getframerate()
    fragment_length = w.getnframes()
    block_length = int(fragment_length * frame_rate / 1000)
    samples = np.frombuffer(w.readframes(w.getnframes()), np.dtype(np.int16))
    samples = (4000 * (samples - np.mean(samples)) / np.std(samples)).astype('int16')
    b, a = butter(16, 4000 / (frame_rate / 2))
    samples = filtfilt(b, a, samples).astype('int16')
    
    weights_size = 1000
    weights = np.repeat([1.0], weights_size) / weights_size

    silence_threshold = 600
    smooth = np.convolve(np.abs(samples), weights, 'same')

    samples = samples[smooth > silence_threshold]   
    return samples


for filename in glob.glob("D:\\mmad\\kurs\\*.wav"):
    w = wave.open(filename, 'rb')
    wavs_train.append(pre_proc(w))

for name in glob.glob("D:\\mmad\\kurs\\rasp\\*.wav"):
    w = wave.open(name, 'rb')
    wavs_test.append(pre_proc(w))  

        
ampl = np.array(wavs_train)
ampl_rasp = np.array(wavs_test)

np.save("1.npy", ampl)
np.save("2.npy",ampl_rasp)


