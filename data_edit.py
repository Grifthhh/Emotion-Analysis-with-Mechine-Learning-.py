# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:32:47 2019

@author: Ugur
"""

import pandas as pd
import numpy as np
import datetime
#%%
data=pd.read_csv("main_data.csv")
#%%
data.drop_duplicates(subset=None, keep="first", inplace=True)
data["Rate"]=[each.replace("%","") for each in data.Rate]
data.Rate=[int(each)/20 for each in data.Rate]

data.Date=[each.split(",")[0] for each in data.Date]
data["Rate State"]=["Olumsuz" if each<4 else "Olumlu" for each in data.Rate]

#%%
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(data["Rate State"])
plt.title("Rate State",color="green")
plt.show()

#%%
data.Date=[each.replace("Şubat","02") for each in data.Date]
data.Date=[each.replace("Åžubat","02") for each in data.Date]
data.Date=[each.replace("Ocak","01") for each in data.Date]
data.Date=[each.replace("Mart","03") for each in data.Date]
data.Date=[each.replace("Nisan","04") for each in data.Date]
data.Date=[each.replace("Mayıs","05") for each in data.Date]
data.Date=[each.replace("MayÄ±s","05") for each in data.Date]
data.Date=[each.replace("Haziran","06") for each in data.Date]
data.Date=[each.replace("Temmuz","07") for each in data.Date]
data.Date=[each.replace("Ağustos","08") for each in data.Date]
data.Date=[each.replace("AÄŸustos","08") for each in data.Date]
data.Date=[each.replace("Eylül","09") for each in data.Date]
data.Date=[each.replace("Ekim","10") for each in data.Date]
data.Date=[each.replace("KasÄ±m","11") for each in data.Date]
data.Date=[each.replace("Kasım","11") for each in data.Date]
data.Date=[each.replace("Aralık","12") for each in data.Date]
data.Date=[each.replace("AralÄ±k","12") for each in data.Date]
#%%
data.Date=[datetime.datetime.strptime(each,"%d %m %Y").date() for each in data.Date]
#data.sort_values(by=['Date'], inplace=True, ascending=True)
#%%
data.to_csv("main_data_v2.csv",index=False)
