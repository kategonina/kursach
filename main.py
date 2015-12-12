# -*- coding: utf-8 -*-
import numpy as np
from scikits.talkbox.features import mfcc
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def ceps(X):  # вычисление мел-кепстральных коэффициентов
    ceps, mspec, spec = mfcc(X,nwin=256, nfft=512, fs=44100, nceps=13)
    X=[]
    for i in range(0,13):   # усреднение коэффициентов по каждому из 13 столбцов полученной матрицы
        X.append(np.mean(ceps[:,i]))
    Vx = np.array(X)  # записываем в массив numpy
    return Vx
 
# загружаем амплитуды из бинарных файлов  
train_ampl = np.load("D:\\mmad\\kurs\\train_ampl.npy")
test_ampl = np.load("D:\\mmad\\kurs\\test_ampl.npy")

# находим длины выборок
train_len=len(train_ampl)
test_len=len(test_ampl)

# создание numpy массивов для мел-кепстральных коэффициентов
train_ceps = np.zeros((train_len,13))
test_ceps = np.zeros((test_len,13))

# нахождение коэффициентов
for i in range(0,train_len):
    train_ceps[i]=ceps(train_ampl[i])
    
for i in range(0,test_len):
    test_ceps[i]=ceps(test_ampl[i])

# названия исполнителей     
train_labels = np.array(["the babe rainbow","the babe rainbow","my chemical romance","my chemical romance","queen latifah","queen latifah",
"seeds of iblis","seeds of iblis","sopor aeternus","sopor aeternus","gorgoroth","gorgoroth","tarja turunen","tarja turunen",
"rakhmaninov","rakhmaninov","the keane","the keane","beyonce","beyonce","florence and the machine","florence and the machine",
"MO","MO","mozart","mozart","queen","queen"])
test_labels = np.array(["the babe rainbow","my chemical romance","queen latifah","seeds of iblis","sopor aeternus","gorgoroth","tarja turunen","rakhmaninov",
"the keane","beyonce","florence and the machine","MO","mozart","queen"])

# добавляем к матрицам коэффициентов столбец с классами(названиями исполнителей)
train_ceps = np.hstack((train_ceps, np.atleast_2d(train_labels).T))
test_ceps = np.hstack((test_ceps, np.atleast_2d(test_labels).T))

# перемешиваем исоплнителей в случайном порядке
np.random.shuffle(train_ceps)
np.random.shuffle(test_ceps)
    
s=0 # счетчик для числа верно классифицированных исполнителей 

# построение классификатора
# параметры: число соседей - 2, взвешенное голосование (чем ближе сосед - тем важнее его голос), p=1  - расстояние городских кварталов
model = KNeighborsClassifier(n_neighbors=2,weights='distance',p=1) 
model.fit(train_ceps[:,0:12],train_ceps[:,13])  # обучение классификатора

print("Classifier parameters:")
print(model)

print("\nPredicted artist - Actual artist\n")

# проверка классификатора на основе тестовой выборки
for i in range(0,test_len): 
    predicted =model.predict(test_ceps[i,0:12]) # предсказанное значение
    predicted = predicted[0].astype('string')  # преобразование в строку
    if(np.array_equal(predicted,test_ceps[i,13].astype('string'))):  # сравниваем предсказанное и реальное значение
        s+=1   #если значения совпадают, увеличиваем счетчик числа верно классифицированных исполнителей
    print(predicted,test_ceps[i,13])
    
# точность классификации  
print("\nPrecision accuracy:")
print(float(s)/float(test_len))