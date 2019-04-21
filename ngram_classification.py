# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 08:14:21 2019

@author: ekrembulbul
"""

import pandas as pd
import numpy as np
#%%
com = pd.read_csv('clean_comments.csv')
#%%
#comment = com.comment[5]
#%%
#com_list = comment.split()
#%%
#com_char_list = list(comment)
#%%
#from nltk import ngrams
#%%
#trigram = ngrams(com_char_list, 3)
#%%
#trigram_list_tuple = []
#%%
#for gram in trigram:
#    trigram_list_tuple.append(gram)
#%%
#trigram_list = []
#%%
#for i in trigram_list_tuple:
#    tmp = ''
#    for j in i:
#        tmp += j
#    trigram_list.append(tmp)
#%%
rates = pd.read_csv('rates.csv')
#%%
x = com.comment.iloc[:10000]
y = rates.rate_state.iloc[:10000].values
#%%
from sklearn.feature_extraction.text import CountVectorizer
#%%
cv = CountVectorizer(ngram_range = (3,3), analyzer = 'char', max_features = 1000)
#%%
x = cv.fit_transform(x).toarray()
#%%
feature_names = cv.get_feature_names()
#%%
from sklearn.model_selection import train_test_split
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
#%%
count = 0
for each in y_test:
    if each == 1:
        count += 1
real_score = count / 5000
#%%
from sklearn.svm import SVC
#%%
model = SVC()
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











