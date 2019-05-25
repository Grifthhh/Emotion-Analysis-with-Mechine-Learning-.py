# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:35:26 2019

@author: Ugur
"""

from gensim.models import FastText
import numpy as np 
import pandas as pd
import re
import nltk as nlp
from tqdm import tqdm
import pickle
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier


#%%
data=pd.read_csv("main_data_v2.csv")
data["Rate State"]=[1 if each =="Olumlu" else 0 for each in data["Rate State"]]
#%% --Liste yapısındaki kelimeleri düzenleme
comment_list=[]
for com in data.Comment:
    com= re.sub("[^a-zA-Z0-8ğüşıöçİĞÜŞÖÇ]"," ",com)
    com=com.lower()
    com=nlp.word_tokenize(com)
    comment_list.append(com)   
 #%%
vector_size=250
window=5
#%% Fassttext modeli oluşturma ve diske kaydetme

fasttext_model = 'fasstext.model'
print("Generating Fasttext Vectors...")
start = time.time()
model= FastText(size = vector_size)
model.build_vocab(comment_list)
model.train( comment_list,window= window,min_count = 1, workers =4 , total_examples = model.corpus_count , epochs = model.epochs)

print("Model created in {} seconds",format(time.time() -start))

model.save(fasttext_model)

del model  

#%%
fasttext_model = 'fasstext.model'
model = FastText.load(fasttext_model)

#%% her bir yorumdaki kelimelerin vektör ortalamalarını hesaplama
main_mean_array=[]
mean_vektor = np.zeros((1,250))
with tqdm(total=len(comment_list)) as pbar:
    for comm in comment_list:
        size= len(comm)
        mean_vektor = np.zeros((1,250))
        for word in comm:
            mean_vektor+=model[word]
        mean_vektor=mean_vektor/size
        main_mean_array.append(mean_vektor)
        pbar.update(1)

#%% Ortalamaları hesaplanmış vektörleri diske kaydetme
np.save('Main_Mean_Array.npy',main_mean_array)
#%%
vector_size=250
main_mean_array=np.load('Main_Mean_Array.npy')
main_mean_array=np.reshape(main_mean_array,(len(main_mean_array),vector_size))
#%%-------------------------------------------------------------------------------------------
#%% Classifier için  train modeli edilir
x=main_mean_array
y=data["Rate State"].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

start = time.time()
clf = OneVsRestClassifier(svm.SVC(class_weight='balanced', kernel='linear'))
clf_output = clf.fit(x_train, y_train)
print("Model created in {} seconds",format(time.time() -start))

#Train edilmiş svm modeli diske kaydedlilir
filename = 'classifier_model_SVM1.sav'
pickle.dump(clf_output, open(filename, 'wb'))

#%% Kaydedilen model kullannılmak üzere kaydedilir
filename = 'classifier_model_SVM1.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#test datasının sınıfları tahmin edilir
y_predict=loaded_model.predict(x_test)
#%% test edilecek test datasının gerçek değerlerinin histogramı gösterilir

#%% Eğitilmiş modelin başarısı hesaplanır
print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
#Tahmin edilenler ve gerçek değerler arası sonuçları karşılaştırılırç
mat = confusion_matrix(y_test, y_predict)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False   )
plt.xlabel('true label')
plt.ylabel('predicted label')

#%% Diğer sınıflandırma metodlarıda karşılaştırılarak en yüksek başarılı sınıf seçilir.
models=[GaussianNB(),
       RandomForestClassifier(n_estimators=100),
       KNeighborsClassifier(n_neighbors=5),
       DecisionTreeClassifier(),
       GradientBoostingClassifier(),
       LogisticRegression(multi_class="auto", solver="liblinear"),
       ExtraTreesClassifier(n_estimators=100),
       BaggingClassifier()
       ]

def best_model(models, show_metrics=False):
        print("INFO: Finding Accuracy Best Classifier...", end="\n\n")
        best_clf=None
        best_acc=0
        for clf in models:
            clf.fit(x_train, y_train)
            y_pred=clf.predict(x_test)
            acc=metrics.accuracy_score(y_test, y_pred)
            print(clf.__class__.__name__, end=" ")
            print("Accuracy:{:.3f}".format(acc))

            if best_acc<acc:
                best_acc=acc
                best_clf=clf
                best_y_pred=y_pred
        
        print("Best Classifier:{}".format(best_clf.__class__.__name__))

#        filename = 'Best_Classifier_model_SVM1.sav'
#        pickle.dump(clf_output, open(filename, 'wb'))       

        if show_metrics:
            mat = confusion_matrix(y_test, best_y_pred)
            sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False   )
            plt.xlabel('true label')
            plt.ylabel('predicted label')
#%%
x=main_mean_array
y=data["Rate State"].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
best_model(models,show_metrics=True)



