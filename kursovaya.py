# -*- coding: utf-8 -*-
import numpy as np
from scikits.talkbox.features import mfcc
from scipy.cluster.vq import kmeans, vq
import scipy
from scipy.io import wavfile
import glob

import sys
from numpy import *
import matplotlib.pyplot as plt

reload(sys)  
sys.setdefaultencoding("Cp1252")

wavs = []
wf=[]   

def ceps(X):
    ceps, mspec, spec = mfcc(X)
    num_ceps = len(ceps)
    Y=[]
    Y.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
    Vx = np.array(Y)
    return Vx
#########################
N=128
def frames(X):   #не нужна, т.к. функция mfcc из модуля scikits.takbox.features уже содержит функцию разбиения сигнала на фреймы
    frame=[]
    num_frames=int(X.size//(N/2)-1)
    for i in range(0,num_frames):  #делим запись на фреймы с перекрытием N/2 = 64
        frame.append([])
        for j in range(i,i+N):
            frame[i].append(X[j])
        i+=N/2
    return frame
###########################
    
for filename in glob.glob("D:\\mmad\\kurs\\*.wav"):   #файлы для обучения
    wavs.append(wavfile.read(filename))

for name in glob.glob("D:\\mmad\\kurs\\rasp\\*.wav"):   #файл для распознавания
    wf=wavfile.read(name)

k = len(wavs)
coeff = []
for i in range(0,k):    
    cps = zeros((1,13))
    cps=(ceps(wavs[i][1]))
    ccps = zeros((1,13))
    cps = vstack([cps,ccps])
    cps = cps.T
    codebook, distortion = kmeans(cps, 4)
    #codebook = abs(codebook)
    codebook = codebook[abs(codebook[:,0]).argsort()]
    code, dist = vq(cps, codebook)
    coeff.append(codebook)

names=["putin","zadornov","galkin","malakhov a","malachov g","tretyak","guberniev","gusev","galustyan","svetlakov","maslyakov","kapitsa"
,"drozdov","obama","torvaldts","pugacheva","dontsova","khakamada","gurchenko","fedorova","valeria","kabaeva"]

kps = zeros((1,13))
kps=(ceps(wf[1]))
kkps = zeros((1,13))
kps = vstack([kps,kkps])
kps = kps.T
codeb, distort = kmeans(kps, 4)
#codeb = abs(codeb)
codeb = codeb[abs(codeb[:,0]).argsort()]
cod, dis = vq(kps, codeb)

d = []

for i in range(0,k):
    d.append(abs(coeff[i]-codeb))    

for i in range(0,k):
    d[i] = delete(d[i],(1), axis=1) 

min_d=d[0][0]
ind_d=0
for i in range(0,k):

    for j in range(0,4):
        if (d[i][j] < min_d):
            min_d = d[i][j]
            ind_d = i
print(ind_d, names[ind_d//2 - 1])
               
    