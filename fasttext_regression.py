# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.svm import SVR
import numpy as np 
import pandas as pd
import pickle
import time

#%% veriseti alınır
data=pd.read_csv("main_data_v2.csv")

#%% fasttext ten oluşiturulmuş ortalama vektör bilgileri diskten alınır.
vector_size=250
main_mean_array=np.load('Main_Mean_Array.npy')
main_mean_array=np.reshape(main_mean_array,(len(main_mean_array),vector_size))

#%% train ve test datası ayrılır.
from sklearn.model_selection import train_test_split
x=main_mean_array
y=data["Rate"].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=1)

#%% SVM regresyon modeli oluşturulur ve eğitilir.
svr_rbf = SVR(kernel='rbf')
t0 = time.time()
svr_rbf.fit(x_train, y_train)
t1 = time.time()
time_linear_train = t1-t0
print("Training time: %fs;" % (time_linear_train))

 
#%% Regresyon modeli kaydedilir.
filename = 'regression_model_SVM.sav'
pickle.dump(svr_rbf, open(filename, 'wb'))
del svr_rbf  
#%% Model Diskten yüklenir.
filename = 'regression_model_SVM.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#%% Tahmin yapılır
y_predict = loaded_model.predict(x_test)
#%% Başarı oranı hesaplanır.
from sklearn.metrics import r2_score
r2_score(y_test, y_predict) 

