# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:42:41 2019

@author: Ugur
"""
import pandas as pd

data=pd.read_csv("main_data_v2.csv")
data=pd.concat([data["Rate State"],data.Comment,data.Rate],axis=1)
data.dropna(axis=0,inplace=True)
data["Rate State"]=[1 if each =="Olumlu" else 0 for each in data["Rate State"]]

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
# f = open('data', 'wb')
# f.write(sparce_matrix)
# f.close()
# =============================================================================
    
#%%

y=data.iloc[:,0].values
x=sparce_matrix

#%%

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,
                                                 random_state=42)

#%%

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)

#%%

y_pred=nb.predict(x_test)
print("aaccuracy:" ,nb.score(y_pred.reshape(-1,1),y_test))

#%%

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)

#%%

 y_pred=rf.predict(x_test)
print("aaccuracy:" ,rf.score(y_pred.reshape(-1,1),y_test))