# -*- coding: utf-8 -*-
"""
Created on Tue May 14 19:44:47 2019

@author: vefa
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 14 19:44:47 2019
@author: vefa
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
#%%data=pd.read_csv("main_data_v2.csv")
data=pd.read_csv("main_data_v2.csv")
data["Rate State"]=[1 if each =="Olumlu" else 0 for each in data["Rate State"]]
#%%
list_category=[]
list_category=[str(each) for each in data["Category0"].unique()]
list_category1=[]
list_category1=[str(each) for each in data["Category1"].unique()]
list_category2=[]
list_category2=[str(each) for each in data["Category2"].unique()]
list_category3=[]
list_category3=[str(each) for each in data["Category3"].unique()]
#%%
main_mean_array=np.load('Main_Mean_Array.npy')
vector_size=250
main_mean_array=np.reshape(main_mean_array,(len(main_mean_array),vector_size))
ngram_array=np.load('ngram_array_x.npy')
ngram_array=np.reshape(ngram_array,(len(ngram_array),3000))
#%%
def RF(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc=metrics.accuracy_score(y_test, y_pred)
    print("Ürün içi başarı:{:.3f}".format(acc))
 #   mat = confusion_matrix(y_test, y_pred)
 #   sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
 #   plt.xlabel('true label')
 #   plt.ylabel('predicted label')
    
def RF_cross(x,y,x2,y2):
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(x, y)
    y2_pred=clf.predict(x2)
    acc=metrics.accuracy_score(y2, y2_pred)
    print("Ürünler Arası Başarı:{:.3f}".format(acc))
#    mat = confusion_matrix(y2, y2_pred)
#    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False   )
 #   plt.xlabel('true label')
 #   plt.ylabel('predicted label')
    
def choiceProduct():
    urun1, urun2 = random.choices(list_category3, k=2)
    df1=data[data.Category3==urun1]
    df2=data[data.Category3==urun2]
    return urun1, urun2, df1, df2

def RFs(df1, df2):
    print("\n\n Fasttext Model\n")  
    print('Urun 1 -> ' + df1.Category0.iloc[0] + '-' + df1.Category1.iloc[0] + '-' + df1.Category2.iloc[0] + '-' + df1.Category3.iloc[0])
    print('Urun 2 -> ' + df2.Category0.iloc[0] + '-' + df2.Category1.iloc[0] + '-' + df2.Category2.iloc[0] + '-' + df2.Category3.iloc[0])
    x1=[main_mean_array[each] for each in df1.index]
    x1=np.array(x1)
    y1=df1["Rate State"].values
    print('Urun 1 ->', end=' ')
    RF(x1, y1)
    x2=[main_mean_array[each] for each in df2.index]
    x2=np.array(x2)
    y2=df2["Rate State"].values
    print('Urun 2 ->', end=' ')
    RF(x2, y2)
    print('Urun 1 -> fit, Urun 2 -> predict', end=' ')
    RF_cross(x1, y1, x2, y2)
    print('Urun 1 -> predict, Urun 2 -> fit', end=' ')
    RF_cross(x2, y2, x1, y1)
    
    print("\n\n 3-gram Model \n")  
    x1=[ngram_array[each] for each in df1.index]
    x1=np.array(x1)
    y1=df1["Rate State"].values
    print('Urun 1 ->', end=' ')
    RF(x1, y1)
    x2=[ngram_array[each] for each in df2.index]
    x2=np.array(x2)
    y2=df2["Rate State"].values
    print('Urun 2 ->', end=' ')
    RF(x2, y2)
    print('Urun 1 -> fit, Urun 2 -> predict', end=' ')
    RF_cross(x1, y1, x2, y2)
    print('Urun 1 -> predict, Urun 2 -> fit', end=' ')
    RF_cross(x2, y2, x1, y1)
#%%
print('- Catagory 3 -')
for each in range(5):
    urun1, urun2, df1, df2 = choiceProduct()
    while (urun1 == 'nan' or urun2 == 'nan') or urun1 == urun2 or df1.Category2.iloc[0] != df2.Category2.iloc[0]:
        urun1, urun2, df1, df2 = choiceProduct()
    print('')
    RFs(df1, df2)
#%%
print('- Catagory 2 -')
for each in range(5):
    urun1, urun2, df1, df2 = choiceProduct()
    while (urun1 == 'nan' or urun2 == 'nan') or urun1 == urun2 or (df1.Category2.iloc[0] == df2.Category2.iloc[0] or df1.Category1.iloc[0] != df2.Category1.iloc[0]):
        urun1, urun2, df1, df2 = choiceProduct()
    print('')
    RFs(df1, df2)
#%%
print('- Catagory 1 -')
for each in range(5):
    urun1, urun2, df1, df2 = choiceProduct()
    while (urun1 == 'nan' or urun2 == 'nan') or urun1 == urun2 or (df1.Category2.iloc[0] == df2.Category2.iloc[0] or df1.Category1.iloc[0] == df2.Category1.iloc[0] or df1.Category0.iloc[0] != df2.Category0.iloc[0]):
        urun1, urun2, df1, df2 = choiceProduct()
    print('')
    RFs(df1, df2)
#%%
print('- Catagory 0 -')
for each in range(5):
    urun1, urun2, df1, df2 = choiceProduct()
    while (urun1 == 'nan' or urun2 == 'nan') or urun1 == urun2 or (df1.Category2.iloc[0] == df2.Category2.iloc[0] or df1.Category1.iloc[0] == df2.Category1.iloc[0] or df1.Category0.iloc[0] == df2.Category0.iloc[0]):
        urun1, urun2, df1, df2 = choiceProduct()
    print('')
    RFs(df1, df2)
#%%

