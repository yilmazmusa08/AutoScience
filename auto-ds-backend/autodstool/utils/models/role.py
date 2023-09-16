import pandas as pd
import numpy as np
import re
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


#CLASS MODEL
class KolonTipiTahmini1:
    def __init__(self, threshold=20):
        self.threshold=threshold
        
    def fit_transform(self, data, columns=None):
        if not isinstance(data, pd.DataFrame):
            data=pd.DataFrame(data)
        if len(data) > 5000:
            data=data.sample(n=5000)
            
        distm=data.apply(lambda x: np.mean(pd.Series(x).value_counts(normalize=True).values))#değişkenin benzersiz değerlerinin frekans dağılımının normalleştirilmiş değerlerinin ortalamasını hesaplar.
        dists=data.apply(lambda x: np.std(pd.Series(x).value_counts(normalize=True).values))# benzersiz değerlerinin frekans dağılımının normalleştirilmiş değerlerinin standart sapmasını hesaplar.
        wc=data.apply(lambda x: round(len(str(x).split()) / len(str(x)), 5))#her bir değişkenin ortalama kelime sayısını hesaplar.
        wcs=data.apply(lambda x: np.std([len(str(kelime).split()) for kelime in x]))#her bir değişkenin kelime sayılarının standart sapmasını hesaplar.
        Len=data.apply(lambda x: len(str(x)) / len(x))#her bir değişkenin ortalamada kaç karakter içerdiğini hesaplar.
        lens=data.apply(lambda x: np.std([len(str(s).replace(" ", "")) for s in x]))# her bir değişkenin karakter sayılarının standart sapmasını hesaplar.
        uniq=data.apply(lambda x: pd.Series(x).nunique())#her bir değişkenin benzersiz değerlerinin sayısını hesaplar.
        most=data.apply(lambda x: pd.Series(x).value_counts().idxmax() if len(x)>0 and not x.isnull().all() else np.nan)#her bir değişkendeki en sık görülen değeri hesaplar.
        mostR=data.apply(lambda x: pd.Series(x).value_counts(normalize=True).values[0] if len(x)>0 and not x.isnull().all() else np.nan)#her bir değişkendeki en sık görülen değerin oranını hesaplar.
        uniqR=data.apply(lambda x: pd.Series(x).nunique() / len(x))# her bir değişkendeki benzersiz değerlerin oranını hesaplar.
        allunique=data.apply(lambda x: int(x.nunique() == len(x)))#her bir değişkendeki değerlerin hepsinin benzersiz olup olmadığını hesaplar.
        kolon_role=[]
        
        if columns:
            data=data[columns]

        for col in data.columns:
            if len(data[col].unique()) == 1:
                data.drop(col, axis=1, inplace=True)
            elif (len(data[col].unique()) == 2):
                kolon_role.append('flag')
            elif len(data[col].unique()) == len(data):
                kolon_role.append('unique')
            elif (all(isinstance(val, int) for val in data[col]) and len(data[col].unique()) == len(data) and set(data[col]) == set(range(1, len(data)+1))):
                kolon_role.append('id')
            elif (all(isinstance(val, int) for val in data[col]) and (re.search(r'(id|ID|Id|iD>|ıd)', col))):
                kolon_role.append('id')
            elif all(isinstance(val, int) for val in data[col]) and len(data[col].unique()) == len(data):
                digits=len(str(data[col].iloc[0]))
                if (data[col].apply(lambda x: len(str(x)) == digits).sum() / len(data[col])) > 0.9:
                    kolon_role.append('id')           
            elif all([isinstance(val, str) and pd.to_datetime(val, errors='coerce') is not pd.NaT for val in data[col].values]) or (data[col].dtype == 'datetime64[ns]'):
                    kolon_role.append('date')
            elif isinstance(data[col].iloc[0], str) and data[col].str.split().str.len().mean() > 4:
                kolon_role.append('text')
            elif len(data[col].unique()) > self.threshold and (all(isinstance(val, int) for val in data[col]) or all(isinstance(val, float) for val in data[col])):
                if data[col].apply(lambda x: (isinstance(x, (int,float)) and len(str(x).split('.')) > 1 and str(x).split('.')[1] != '0' and str(x).split('.')[1] != '0000')).sum() > 0:
                    kolon_role.append('scalar')
                elif data[col].apply(lambda x: isinstance(x, (int))).all():
                    kolon_role.append('numeric')
                else:
                    kolon_role.append('numeric')
            elif isinstance(data[col].iloc[0], str) and len(data[col].unique()) >= 0.9*len(data) and data[col].str.split().str.len().mean()<5:
                kolon_role.append('identifier')     
            elif len(data[col].unique()) > self.threshold and (all(isinstance(val, int) for val in data[col]) or all(isinstance(val, float) for val in data[col])):
                kolon_role.append('numeric')
            elif len(data[col].unique()) < self.threshold:
                kolon_role.append('categoric')
            else:
                kolon_role.append("identifier")

        if 'date' in kolon_role:
            date_cols=[col for col, role in zip(data.columns, kolon_role) if role == 'date']
            for date_col in date_cols:
                col_idx=data.columns.get_loc(date_col)
                kolon_role[col_idx]='date'
        #Eğer kolon değerleri datetime ise o zaman hangı formatda olduğunu gosterir
        def get_Date_Format(data) -> str:
            if not isinstance(data, pd.DataFrame):
                dataframe=pd.DataFrame(data)

            date_formats=[   
                "%m/%Y",
                "%m-%Y",    
                "%d/%m/%y",  
                "%m/%d/%y",  
                "%d.%m.%Y",  
                "%d/%m/%Y", 
                "%m/%d/%Y",  
                "%Y-%m-%d",  
                "%Y/%m/%d",  
                "%m-%d-%Y",  
                "%d-%m-%Y", 
                "%d.%m.%y",
                "%m.%d.%y",
                "%Y/%m",
                "%Y-%m",
                "%m/%d",
                "%d.%m",
                "%d/%m",
                "%m.%d",
                "%Y",
                "%d-%m"
                ]
            dict1={}
            for column in data.columns:
                values=data[column]
                for f in date_formats:
                    try:
                        date=pd.to_datetime(values, format=f)
                        if date.dt.strftime(f).eq(values).all():
                            dict1=f
                            break
                    except ValueError:
                        pass
                else: 
                    dict1=None
            return dict1
        date_formats = data.apply(lambda col: get_Date_Format(col.to_frame()))

        sonuc=[]
        for i in data.columns:
            distm_val=round(distm[i], 5) if distm.get(i) is not None else None
            dists_val=round(dists[i], 5) if dists.get(i) is not None else None
            wc_val=round(wc[i], 5) if wc.get(i) is not None else None
            wcs_val=round(wcs[i], 5) if wcs.get(i) is not None else None
            len_val=round(Len[i], 5) if Len.get(i) is not None else None
            lens_val=round(lens[i], 5) if lens.get(i) is not None else None
            uniq_val=round(uniq[i], 5) if uniq.get(i) is not None else None
            most_val=most[i] if most.get(i) is not None else None
            mostR_val=round(mostR[i], 5) if mostR.get(i) is not None else None
            uniqR_val=round(uniqR[i], 5) if uniqR.get(i) is not None else None
            allunique_val=round(allunique[i], 5) if allunique.get(i) is not None else None
            kolon_role_val=kolon_role[data.columns.get_loc(i)]
            top3=data[i].value_counts().nlargest(3).index.tolist()
            sonuc.append({
                'col_name': i,
                'distm': distm_val,
                'date_formats': date_formats[i],
                'dists': dists_val,
                "wc":wc_val,
                "wcs":wcs_val,
                "len":len_val,
                "lens":lens_val,
                "uniq":uniq_val,
                "uniqR %":uniqR_val,
                "most":most_val,
                "mostR %":mostR_val,
                "allunique":allunique_val,
                "top3":top3,
                'Role': kolon_role_val})

        result={}
        for d in sonuc:
            col_name=d.pop('col_name')
            result[col_name]=d
        return result


# Principal Component
def calculate_pca(df, comp_ratio=1.0, target=None):
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() < 20:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
    if target is not None:
        df = df.drop(columns=[target])
    df.fillna(df.mean(), inplace=True)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    pca = PCA()
    pca.fit(df[numeric_cols])
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum_var_ratio = np.cumsum(explained_var_ratio)
    n_components = np.searchsorted(cumsum_var_ratio, comp_ratio) + 1
    pca = PCA(n_components=n_components)
    pca.fit(df[numeric_cols])
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum_var_ratio = np.cumsum(explained_var_ratio)
    result_dict = {
        'Cumulative Explained Variance Ratio': cumsum_var_ratio.tolist()}
    result_dict['Principal Component'] = list(
        range(1, len(explained_var_ratio)+1))
    return result_dict


def set_to_list(data):
    if isinstance(data, set):
        return list(data)
    if isinstance(data, dict):
        return {set_to_list(key): set_to_list(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [set_to_list(item) for item in data]
    if isinstance(data, np.int64):
        return int(data)
    return data


# df = pd.read_csv("train.csv")
# print(analysis(df,target="Survived",return_stats=True))
