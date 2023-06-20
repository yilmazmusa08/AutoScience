import os
import json
import uvicorn
import warnings
import numpy as np
import pandas as pd
from typing import List
from fastapi import FastAPI, File, UploadFile, APIRouter
from typing import List, Dict
from pydantic import BaseModel
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
from fastapi import FastAPI, Form, Request, Body
from type_pred import *

import sys

# models klasörünün tam yolu
models_path = '/home/firengiz/Belgeler/proje/automl/models'

# models_path'i sys.path listesine ekle
sys.path.append(models_path)
from init import *
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse


class OutputData(BaseModel):
    data: List[dict]


app = FastAPI()

templates = Jinja2Templates(directory="template")

# Define a route for serving static files
app.mount("/dist", StaticFiles(directory="dist"), name="dist")
app.mount("/plugins", StaticFiles(directory="plugins"), name="plugins")
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML page setup
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Render the HTML template
    return templates.TemplateResponse("index.html", {"request": request})


import traceback


@app.post('/', response_class=HTMLResponse)
async def run_model_api(request: Request, file: UploadFile = File(...), target: str = Form(None), date: str = Form(None)):
    df = pd.read_csv(file.file)
    sonuc2 = create_model(df, date=date, target=target)
    sonuc2 = set_to_list(sonuc2)

    output_file2 = os.path.join(os.getcwd(), "output2.json")
    with open(output_file2, "w") as f:
        json.dump(sonuc2, f)

    # '/process' endpointine yönlendirme yap
    return RedirectResponse(url="/process", status_code=302)



@app.get("/process", response_class=HTMLResponse)
async def run_process_model(request: Request):
    try:
        with open("output1.json", "r") as f1, open("output2.json", "r") as f2:
            output_dict2 = json.load(f1)

        # Render the HTML template
        return templates.TemplateResponse("index.html", {"request": request, "result": json.dumps(output_dict2)})
    except:
        return "Dataset kabul edilmedi, bir hata oluştu."


@app.post("/", response_class=HTMLResponse)
async def run_analysis_api(request: Request, file: UploadFile = File(...), target: str = Form(None), warning: bool = Form(True), return_stats: bool = Form(False), comp_ratio: float = Form(1.0)):
    try:
        df = pd.read_csv(file.file)

        sonuc = analysis(df, target=target, warning=warning, return_stats=return_stats) if target else analysis(df, target=None, warning=warning, return_stats=return_stats)

        pca_dict = {}
        for col in sonuc['Role']:
            null_counts = df.isnull().sum()
            empty_cols = null_counts[null_counts >= len(df) * 0.6].index
            df.drop(empty_cols, axis=1, inplace=True)

            for col in df.columns:
                if df[col].dtype == 'object' and df[col].nunique() < 20:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                elif df[col].dtype != 'object':
                    df[col].fillna(df[col].mean(), inplace=True)

            df[col] = df[col].fillna(df[col].mean())
            result_dict = calculate_pca(df.select_dtypes(include=['float', 'int']), comp_ratio=comp_ratio)
            pca_dict = {
                'Cumulative Explained Variance Ratio': result_dict['Cumulative Explained Variance Ratio'],
                'Principal Component': result_dict['Principal Component']
            }

        sonuc['PCA'] = pca_dict
        sonuc = set_to_list(sonuc)
        output_file = os.path.join(os.getcwd(), "output.json")

        with open(output_file, "w") as f:
            json.dump(sonuc, f)

        # Redirect to the '/process' endpoint
        return RedirectResponse(url="/process", status_code=302)
    except Exception as e:
        traceback.print_exc()  # Hata mesajını terminale yazdır
        return "Dataset kabul edilmedi, bir hata oluştu."


@app.get("/process", response_class=HTMLResponse)
async def run_process(request: Request):
    try:
        with open("output.json", "r") as f:
            output_dict = json.load(f)

        # Render the HTML template
        return templates.TemplateResponse("index.html", {"request": request, "result": json.dumps(output_dict)})
    except:
        return "Dataset kabul edilmedi, bir hata oluştu."


"""@app.post("/model")
async def process_csv(file: UploadFile = File(...), target: str = None):
    # Yüklenen CSV dosyasını oku
    df = pd.read_csv(file.file)

    # Verileri ön işleme fonksiyonunu çağır
    df = preprocess(df)

    # Model oluşturma fonksiyonunu çağır
    result = create_model(df, target=target)

    return result"""


@app.get("/")
def Description():
    """
    # Programın amacı, veri biliminde veri ön işleme işlemlerinin otomatikleştirilmesine yardımcı olmaktır.💻 💡
    Bu program veri ön işleme adımlarının bir kısmını otomatik hale getirerek veri biliminde çalışanlara zaman kazandırmayı hedeflemektedir.

    Veri ön işleme işlemlerinin bir kısmını otomatik hale getirerek veri bilimi alanında 
    çalışanlara zaman kazandırmak, daha hızlı ve doğru sonuçlar elde etmelerine yardımcı olmaktır. Bu otomasyon, veri bilimi uzmanlarına daha fazla
    zaman kazandırarak veri analizine daha fazla odaklanmalarını sağlayabilir.

    Bu fonksiyon, bir veri setinin analiz edilmesi için tasarlanmıştır. Fonksiyon, veri setindeki yüksek korelasyona sahip sütunları ve 
    hedef değişkenle yüksek korelasyona sahip sütunları belirleyebilir. Fonksiyon ayrıca her sütunun hangi dağılıma sahip olduğunu tahmin edebiliyor.
    Bu özelliklerin yanı sıra, fonksiyon, veri seti istatistiklerini içeren bir sözlük döndürüyor.

    # Fonksiyonun parametreleri:
    #
            df: analiz edilecek veri seti
            target: hedef değişkeni belirleyen bir boolean (varsayılan olarak None)
            return_stats: veri seti istatistiklerini içeren bir sözlük döndürmek için bir boolean (varsayılan olarak False)

    Kod, sütunlardaki benzersiz değerlerin frekans dağılımının normalleştiriilmiş değerlerinin ortalaması, standart sapması, ortalama kelime sayısı,
    kelime sayılarının standart sapması, karakter sayısının ortalaması, karakter sayılarının standart sapması, benzersiz değerlerin sayısı, en sık görülen değer, 
    en sık görülen değerin oranı, benzersiz değerlerin oranı ve değerlerin benzersiz olup olmadığı gibi özelliklerini hesaplar.

    # Kod nasıl çalışır ?...
    # 
            1) df,target = ...( target yazılırsa:target'i tırnak içinde yazmaya gerek yok. ), return_stats=True --> Role, Warning,high_corr_target
               feature_importance, distributions, prblem_type
            2) df,target = None, return_stats=True --> Role, Warning, high_corr_target, prblem_type
            Not: Eğer target=None ise o zaman  high_corr_target ve feature_importance sonuçlarını vermez ( targete'e bağlı çalışıyorlar )
                 target = None,Send empty value tik'e tıklamanız yerterli olucaktır 
    #
            3) df,target = None, return_stats=False --> Role, Warning, distributions, high_corr_target, prblem_type
            4) df,target = None, return_stats=False --> Role, Warning, distributions, high_corr_target, feature_importance, prblem_type
    """
    return Description

uvicorn.run(app, host="127.0.0.1", port=5050)

# kodu durdurmakıcın bu kodu yaz: sudo lsof -i :8000    sudo kill 12345(buraya  durdurmak ıstedıgın  PID   numarasını yaz)
