import seaborn as sns
import pandas as pd
import random
netflix=pd.read_csv('netflix_titles.csv.zip')
netflix



netflix.info()
#nan degerlerden kurtulma
netflix.dropna(inplace=True)

netflix.fillna('-',inplace=True)#atmak yerine nan gecen yerleri her hangi bir sey ile doldura bilirsiniz


netflix['director'].replace('-','yonetmen bilinmiyor',inplace=True)# tre olan yerleri yonetmen bilinmiyor diye yazdirdik
netflix.sort_values('release_year',inplace=True)#release_year colonunu kucukten buyuge siraladik
netflix.sort_values('director',na_position='last',inplace=True)#na_position yaparsak nan degerleerni son atar
matrix=netflix.values
matrix

netflix.rename(columns={'show_id':'Tv Show icin benzersiz kimlik',
                        'type':'Filim ve ya TV show',
                        'title':'Filmin adi/TV show',
                        'director':'Filmin yonetmeni',
                        'cast':'Filmde/Show yer alan aktorler',
                        'country':'Filmin/gosterinin uretildigi ulke',
                        'date_added':'Filmin Eklendigi  Tarih',
                        'release_year':'Gosterinin Gercek Yayin Yili',
                        'rating':'Film/Tv Show derecelendirilmesi',
                        'duration':'Toplam sure-dakika ve ya sezon sayisi'},inplace=True)




liste=[]
for i in range(1,5333):
    liste.append(random.randint(1,5))
sonuc2=liste
sonuc2

netflix['rating__']=liste
netflix
sayac=netflix['Filmin/gosterinin uretildigi ulke'][netflix['Filmin/gosterinin uretildigi ulke']=='India'].count()
sayac


netflix['ulke_sonuc'][netflix['Filmin/gosterinin uretildigi ulke']=='India']=netflix['rating__'].sum()//sayac


netflix.insert(loc=6,column='ulke_sonuc',value=0)
netflix

grafik=netflix[['rating__','Filmin/gosterinin uretildigi ulke']]
grafik


sns.catplot(data=netflix, x="Filim ve ya TV show", y="Gosterinin Gercek Yayin Yili")

