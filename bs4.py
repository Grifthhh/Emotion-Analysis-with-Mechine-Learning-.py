from proje_v2 import takeComment
from bs4 import BeautifulSoup
import requests
import pandas as pd

#%%

def decision(url):
    head_pr={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36"}
    r = requests.get(url, headers=head_pr)
    soup = BeautifulSoup(r.content,"lxml")
    a = list(soup.find_all(class_='allReviews'))
    
    if len(a) != 0:
        b = a[0].text
        
        word = b.split()
        return int(word[2][word[2].find('(') + 1 : word[2].find(')')])
    else:
        return 0
    
#%%

link_list = []
url_list = ['https://www.hepsiburada.com/yapi-market-bahce-oto-c-60002705?siralama=yorumsayisi',
            'https://www.hepsiburada.com/ev-dekorasyon-c-60002028?siralama=yorumsayisi',
            'https://www.hepsiburada.com/pet-shop-c-2147483616?siralama=yorumsayisi']

#%%

for j in url_list:
    for i in range(1):
        head_pr={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36"}
        r = requests.get(j + '&sayfa=' + str(i), headers=head_pr)
        soup = BeautifulSoup(r.content,"lxml")
        page=soup.find('div',attrs={'class':'pagination'})
        link = soup.find_all('div',attrs={'class':'box product no-hover hb-placeholder'})
        for x in link:
            url = 'https://www.hepsiburada.com' + x.a.get('href')
            if decision(url) > 20:
                link_list.append(url + '-yorumlari')
        link = soup.find_all('div',attrs={'class':'box product hb-placeholder'})
        for x in link:
            url = 'https://www.hepsiburada.com' + x.a.get('href')
            if decision(url) > 20:
                link_list.append(url + '-yorumlari')

#%%
                
dict={'Comment':[],"Subject":[],"Date":'',"User":[],"Rate":[],"Comment-Like":[],"Comment-Dislike":[], "Category0": [], "Category1": [], "Category2": [], "Category3": []}
data=pd.DataFrame(dict)

#%%

len(link_list)
link_list
for x in link_list:
    data = pd.concat([data, takeComment(x)], axis = 0, ignore_index=True)
    
#%%

data
data.to_csv('data015.csv', index = False)





























