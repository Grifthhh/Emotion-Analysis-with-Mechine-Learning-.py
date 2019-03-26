from bs4 import BeautifulSoup
import requests
import pandas as pd

#%%

def takeComment(url):
    head_pr={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36"}
    r = requests.get(url,headers=head_pr)
    soup = BeautifulSoup(r.content,"lxml")
    page=soup.find('div',attrs={'class':'pagination'})

    yorum_list=[]
    baslik_list=[]
    tarih_list=[]
    kullanici_list=[]
    rate_list=[]
    yes_list=[]
    no_list=[]
    page_url = url

    for i in range(1,(int(page.select('a')[-1].text))+1):
        url=page_url+'?sayfa='+str(i)
        r = requests.get(url,headers=head_pr)
        soup = BeautifulSoup(r.content,"lxml")

        yorum=soup.find_all('p',attrs={'class':'review-text'})
        baslik=soup.find_all('strong',attrs={'class':'subject'})
        tarih=soup.find_all('strong',attrs={'class':'date'})
        kullanici=soup.find_all('span',attrs={'class':'user-info'})
        
        ratings=soup.find_all('div',attrs={'class':'ratings active'})
        rates=[]
        for i in range(6,len(ratings)):
            rates.append(ratings[i])
        
        yes=soup.find_all('a',attrs={'class':'yes'})
        no=soup.find_all('a',attrs={'class':'no'})
        s=soup.find_all('span',attrs={'itemprop':'title'})

        for x in yorum:
            yorum_list.append(x.text)
        for x in baslik:
            baslik_list.append(x.text)
        for x in tarih:
            tarih_list.append(x.text)
        for x in kullanici:
            kullanici_list.append(x.text)
        for x in rates:
            rate_list.append(str(x).split('"')[3].split(' ')[1])
        for x in yes:
            yes_list.append(x.b.text)
        for x in no:
            no_list.append(x.b.text)

    dict={'Comment':yorum_list,"Subject":baslik_list,"Date":tarih_list,"User":kullanici_list,"Rate":rate_list,"Comment-Like":yes_list,"Comment-Dislike":no_list}
    data=pd.DataFrame(dict)
    for i in range(4):
        for j in baslik_list:
            data['Category' + str(i)] = ''
    i = 0
    
    for each in s:
        if i < 4:
            data['Category' + str(i)]=each.text
            i = i + 1
    return data










































