from role import *
from prep_tools import *
import pandas as pd


df = pd.read_excel("Online_Retail.xlsx")

def preprocess(df):
    kt = KolonTipiTahmini1()  # KolonTipiTahmini1 sınıfı veya modülünün doğru şekilde yüklendiğinden emin olun

    data_dict = {}
    delete_cols = []  # Silinecek sütunları depolamak için boş bir liste oluşturun

    for col in df.columns:
        data_dict[col] = kt.fit_transform(df[[col]])[col]

        if "Role" in data_dict[col] and data_dict[col]["Role"] == "identifier":
            delete_cols.append(col)  # Silinecek sütunları listeye ekleyin
    print(delete_cols)
    df = df.drop(columns=delete_cols)  # Silinecek sütunları DataFrame'den kaldırın
    print("DELETED DATAFRAME : ", df)
    return df, data_dict

obj = list(df.select_dtypes(include=['object']).columns)
num = list(df.select_dtypes(exclude=['object']).columns)
for col in df:
    df[col] = df.apply(lambda row: fill_na(row, col, obj=obj, num=num), axis = 1)

df, result = preprocess(df)

print(result)
df.info()