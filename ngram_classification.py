# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 08:14:21 2019

@author: ekrembulbul
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
import pickle
#%%
comment_start = 0
comment_end = 50000
matrix_size = 5000
#%% Diğer sınıflandırma metodlarıda karşılaştırılarak en yüksek başarılı sınıf seçilir.
models=[GaussianNB(),
       RandomForestClassifier(n_estimators=100),
       KNeighborsClassifier(n_neighbors=5),
       DecisionTreeClassifier(),
       SVC(gamma='scale'),
       GradientBoostingClassifier(),
       LogisticRegression(multi_class="auto", solver="liblinear"),
       ExtraTreesClassifier(n_estimators=100),
       BaggingClassifier()]

def best_model(models, show_metrics=False):
        print("INFO: Finding Accuracy Best Classifier...", end="\n\n")
        best_clf=None
        best_acc=0
        for clf in models:
            clf.fit(x_train, y_train)
            y_pred=clf.predict(x_test)
            acc=metrics.accuracy_score(y_test, y_pred)
            print(clf.__class__.__name__, end=" ")
            print("Accuracy: {:.3f}".format(acc))

            if best_acc<acc:
                best_acc=acc
                best_clf=clf
                best_y_pred=y_pred
            
            fileName = clf.__class__.__name__ + '_' + str(comment_start) + '-' + str(comment_end) + '.sav'
            pickle.dump(clf, open(fileName, 'wb'))
        
        print("\nBest Classifier: {}".format(best_clf.__class__.__name__))

#        filename = 'Best_Classifier_model_SVM1.sav'
#        pickle.dump(clf_output, open(filename, 'wb'))       

        if show_metrics:
            mat = confusion_matrix(y_test, best_y_pred)
            sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False   )
            plt.xlabel('true label')
            plt.ylabel('predicted label')
#%%
com = pd.read_csv('clean_comments.csv')
#%%
#comment = com.comment[5]
#com_list = comment.split()
#com_char_list = list(comment)
#from nltk import ngrams
#trigram = ngrams(com_char_list, 3)
#trigram_list_tuple = []
#for gram in trigram:
#    trigram_list_tuple.append(gram)
#trigram_list = []
#for i in trigram_list_tuple:
#    tmp = ''
#    for j in i:
#        tmp += j
#    trigram_list.append(tmp)
#%%
rates = pd.read_csv('rates.csv')
x = com.comment.iloc[comment_start:comment_end]
y = rates.rate_state.iloc[comment_start:comment_end].values
#%%
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range = (3,3), analyzer = 'char', max_features=matrix_size)
#cv = CountVectorizer(ngram_range = (3,3), max_features = 1000)
x = cv.fit_transform(x).toarray()
feature_names = cv.get_feature_names()
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

count = 0
for each in y_test:
    if each == 1:
        count += 1
real_score = count / len(y_test)
print('\nComment Size: {0} - {1}'.format(comment_start, comment_end))
print('Matrix Size: {}'.format(matrix_size))
print('Positive Negative Comment Rate: {}\n'.format(real_score))

best_model(models,show_metrics=True)
#%%
#model = SVC(probability = True, class_weight = 'balanced', kernel = 'linear')
#model = SVC(kernel = 'linear')
#model.fit(x_train, y_train)
#score = model.score(x_test, y_test)
#y_pred = model.predict(x_test)
#%%
#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(y_test, y_pred)











