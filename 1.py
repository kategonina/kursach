# -*- coding: utf-8 -*-
import numpy as np
from scikits.talkbox.features import mfcc
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def ceps(X):
    ceps, mspec, spec = mfcc(X,nwin=256, nfft=512, fs=44100, nceps=13)
    X=[]
    #num_ceps = len(ceps)
    for i in range(0,13):
        X.append(np.mean(ceps[:,i]))
    #X.append(np.mean(ceps[int(num_ceps*1/10):(num_ceps*9/10)],axis=0))
    Vx = np.array(X)
    return Vx
    
ampl = np.load("D:\\mmad\\kurs\\1.npy")
ampl_rasp = np.load("D:\\mmad\\kurs\\2.npy")

k=len(ampl)

cps = np.zeros((k,13))
kps = np.zeros((k/2,13))

for i in range(0,k):
    cps[i]=ceps(ampl[i])
    
for i in range(0,k/2):
    kps[i]=ceps(ampl_rasp[i])
    

#np.save("D:\\mmad\\kurs\\5.npy",cps,delimiter=" ", fmt="%s")
#np.save("D:\\mmad\\kurs\\6.npy",kps,delimiter=" ", fmt="%s")

model = KNeighborsClassifier(n_neighbors=3,weights='distance',p=1)
model.fit(cps,[1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22])
print(model)
for i in range(0,k/2): 
    print(model.predict(kps[i]))