# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:53:53 2019

@author: EKREMBÜLBÜL
"""

#%%
import pandas as pd
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











