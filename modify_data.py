# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:53:53 2019

@author: EKREMBÜLBÜL
"""

#%%
import pandas as pd
import numpy as np
import re
#%%
comments = pd.read_csv("comments.csv")
#%%
comm = []
for com in comments.Comment:
    com = re.sub("[^a-zA-Z0-8ğüşıöçİĞÜŞÖÇ]"," ",com)
    com = com.lower()
    com = com.split()
    com = " ".join(com)
    comm.append(com)
#%%
comm = pd.DataFrame(comm)
#%%
comm.to_csv("clean_comments.csv", index = False)
#%%
com = pd.read_csv("clean_comments.csv")
#%%
data = pd.read_csv('main_data_v2.csv')
#%%
rate = list(data.Rate)
rate_int = []
#%%
for each in rate:
    rate_int.append(int(each))
#%%
rate_state = list(data['Rate State'])
#%%
rate_state_bool = [1 if each == 'Olumlu' else 0 for each in rate_state]
#%%
data = pd.DataFrame
#%%
data = {'rate': rate_int, 'rate_state': rate_state_bool}
#%%
df = pd.DataFrame(data)
#%%
df.to_csv("rates.csv", index = False)










