# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:26:22 2019

@author: ekrembulbul
"""

import pandas as pd
#%%
com = pd.read_csv('clean_comments.csv')
#%%
rates = pd.read_csv('rates.csv')
#%%
x = com.comment
y = rates.rate.values
#%%
from sklearn.feature_extraction.text import CountVectorizer
#%%
cv = CountVectorizer(ngram_range = (3,3), analyzer = 'char', max_features = 2000)
#%%
x = cv.fit_transform(x).toarray()
#%%
feature_names = cv.get_feature_names()
#%%
from sklearn.model_selection import train_test_split
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
from sklearn.svm import SVR
#%%
model = SVR(gamma = 0.01, C = 100, probability = True, class_weight = 'balanced', kernel = 'linear')
#model = SVR()
#%%
model.fit(x_train, y_train)
#%%
score = model.score(x_test, y_test)
#%%
y_pred = model.predict(x_test)
#%%
from sklearn.metrics import accuracy_score
#%%
accuracy = accuracy_score(y_test, y_pred)



















