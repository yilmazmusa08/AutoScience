import os
import json
import uvicorn
import warnings
import pandas as pd
from typing import List
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
from fastapi import FastAPI, Form, Request
from type_pred import *
from sklearn.exceptions import ConvergenceWarning
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_login.exceptions import InvalidCredentialsException
from fastapi_login import LoginManager

SECRET = "super-secret-key"
manager = LoginManager(
    SECRET, '/login',
    use_cookie=True
)

engine = create_engine('sqlite:///database.db', echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String)
    password = Column(String)

Base.metadata.create_all(engine)

def save_user_to_database(username: str, password: str) -> int:
    # Yeni bir oturum oluştur
    session = Session()

    try:
        # Kullanıcıyı oluştur ve veritabanına ekle
        user = User(username=username, password=password)
        session.add(user)
        session.commit()

        return user.id
    except Exception as e:
        # Hata durumunda geri alma işlemleri yapılabilir
        print("Database Error:", e)
        session.rollback()
        return -1
    finally:
        # Oturumu kapat
        session.close()

def validate_user(username: str, password: str) -> bool:
    # Yeni bir oturum oluştur
    session = Session()

    try:
        # Kullanıcıyı veritabanında ara
        user = session.query(User).filter_by(username=username, password=password).first()

        if user:
            return True
        else:
            return False
    except Exception as e:
        print("Database Error:", e)
        return False
    finally:
        session.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    # Erişim tokenını kullanarak mevcut kullanıcıyı doğrulayın
    # Örneğin, JWT (JSON Web Token) kullanarak erişim tokenını doğrulayabilirsiniz
    # Doğrulama işlemlerini gerçekleştirin ve kullanıcıyı döndürün
    # Bu örnekte, basitçe kullanıcı adını token olarak kabul ediyoruz
    user = User(username=token, password="")
    return user


class OutputData(BaseModel):
    data: List[dict]

def is_user_subscribed(user: User) -> bool:
    # Kullanıcının abone olup olmadığını kontrol etmek için gereken işlemleri burada gerçekleştirin
    # Örneğin, kullanıcının abonelik durumunu bir veritabanı sorgusuyla kontrol edebilirsiniz
    # Eğer kullanıcı abonelik durumuna sahipse True değerini döndürün, değilse False değerini döndürün

    # Örnek bir kontrol:
    # Burada kullanıcının abonelik durumunu kontrol eden bir sorgu yapılıyor
    # Sizin veritabanı yapınıza ve gereksinimlerinize göre bu sorguyu uyarlamalısınız
    session = Session()
    try:
        subscribed_user = session.query(User).filter_by(id=user.id, is_subscribed=True).first()

        if subscribed_user:
            return True
        else:
            return False
    except Exception as e:
        print("Database Error:", e)
        return False
    finally:
        session.close()


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

@app.post("/register")
async def register_user(request: Request):
    form_data = await request.form()
    username = form_data.get("username")
    password = form_data.get("password")

    if username and password:
        user_id = save_user_to_database(username, password)
        if user_id != -1:
            return {"message": "User registered successfully!"}
        else:
            return {"message": "Failed to register user."}
    else:
        return {"message": "Invalid username or password."}

@app.post("/login")
def login(username: str, password: str):
    # Kullanıcıyı veritabanında doğrulama işlemleri burada gerçekleştirilir
    # Örneğin, kullanıcı adı ve parolanın doğru olup olmadığını kontrol edebilirsiniz
    # Eğer doğruysa, kullanıcının oturumunu başlatın ve bir erişim tokenı döndürün
    # Eğer yanlışsa, HTTPException kullanarak hata döndürün
    if validate_user(username, password):
        access_token = manager.create_access_token(username)
        return {"access_token": access_token}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get('/protected')
def protected_route(user=Depends(manager.optional)):
    if user is None:
        return {"message": "User not found."}
        # Do something ...
    return {'user': user}


@app.post("/", response_class=HTMLResponse)
async def run_analysis_api(request: Request, file: UploadFile = File(...), target: str = Form(None), warning: bool = Form(True), return_stats: bool = Form(False), comp_ratio: float = Form(1.0), current_user: User = Depends(get_current_user)):
    if not is_user_subscribed(current_user):
        raise HTTPException(status_code=403, detail="Not subscribed")
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


@app.get("/process", response_class=HTMLResponse)
async def run_process(request: Request):
    with open("output.json", "r") as f:
        output_dict = json.load(f)

    # Render the HTML template
    return templates.TemplateResponse("index.html", {"request": request, "result": json.dumps(output_dict)})



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
