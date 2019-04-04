# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:33:06 2019

@author: EKREMBÜLBÜL
"""

import pandas as pd

data=pd.read_csv("main_data_v2.csv")
data=pd.concat([data["Rate State"],data.Comment,data.Rate],axis=1)
data.dropna(axis=0,inplace=True)
data["Rate State"]=[1 if each =="Olumlu" else 0 for each in data["Rate State"]]

#%%

#data = data.iloc[:20000, :]

#%%

import re
import nltk as nlp
nlp.download("stopwords")
nlp.download('punkt')
from nltk.corpus import stopwords

#%%

from TurkishStemmer import TurkishStemmer
comment_list=[]
for com in data.Comment:
    com= re.sub("[^a-zA-Z0-8ğüşıöçİĞÜŞÖÇ]"," ",com)
    com=com.lower()
#   com=nlp.word_tokenize(com)
#   stemmer=TurkishStemmer()
#   com=[stemmer.stem(word) for word in com]
#   com=" ".join(com)
    comment_list.append(com)
    
#%%
    
from sklearn.feature_extraction.text import CountVectorizer

max_feature=1000
count_vectorizer=CountVectorizer(max_features=max_feature)
sparce_matrix=count_vectorizer.fit_transform(comment_list).toarray()

#%%

# =============================================================================
# count = 0
# deleteList = []
# size = sparce_matrix[:, 0].size
# for i in range(0, size):
#     for j in sparce_matrix[i, :]:
#         count += j;
#     if count == 0:
#         deleteList.append(i)
#     count = 0
# 
# import numpy as np
# 
# for i in deleteList:
#     np.delete(sparce_matrix, i)    
# =============================================================================

#%%

y=data.iloc[:,2].values.reshape(-1, 1)
x=sparce_matrix

#%%

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,
                                                 random_state=42)

#%%
import numpy as np
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(x_train, y_train.ravel())

#%%

# y_pred=rf.predict(x_test)
# #y_test=y_test.reshape(100)
# y_pred = y_pred.reshape(-1, 1)
#print("aaccuracy:" ,rf.score(y_pred,y_test))

from sklearn.metrics import r2_score
y_pred = rf.predict(x_test)
y_pred = y_pred.reshape(-1, 1)
print('accuracy:', r2_score(y_test, y_pred.reshape(-1, 1)))




















