# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:44:35 2019

@author: EKREMBÜLBÜL
"""

import numpy as np
import pandas as pd

com = pd.read_csv('clean_comments.csv')
rates = pd.read_csv('rates.csv')

x = com.comment
y = rates.rate_state.values

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range = (3,3), analyzer = 'char', max_features=3000)
x = cv.fit_transform(x).toarray()

np.save('ngram_array_x.npy',x)











