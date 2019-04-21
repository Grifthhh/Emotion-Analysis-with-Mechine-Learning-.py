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
x = com.comment.iloc[:10000]
y = rates.rate.iloc[:10000]
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



















