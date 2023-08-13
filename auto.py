import os
import sys
import json
import uvicorn
import warnings
import traceback
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, Request, Request, Depends, HTTPException
from sklearn.exceptions import ConvergenceWarning
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from fastapi_login import LoginManager
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
import tempfile

path = "./models"
sys.path.append(path)
from init import *

app = FastAPI()

SECRET = "supersecret"
manager = LoginManager(SECRET, "/login")

# Veritabanı bağlantısı oluşturma
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create a password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def update_passwords():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        for user in users:
            if not pwd_context.identify(user.password):
                # If the password is not bcrypt hashed, update it
                hashed_password = pwd_context.hash(user.password)
                user.password = hashed_password
        db.commit()
        print("Password update successful")
    except Exception as e:
        print("Password update failed:", str(e))
    finally:
        db.close()

update_passwords()

# Upload Templates
templates = Jinja2Templates(directory="template")

# Mounting Static Files
app.mount("/dist", StaticFiles(directory="dist"), name="dist")
app.mount("/plugins", StaticFiles(directory="plugins"), name="plugins")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(
    request: Request,
    signup_success: bool = False,
    message: str = "",
    email: str = ""
):
    is_logged_in = bool(email)
    logo_button = (
        f'<a class="nav-link" href="#" style="display: block;">'
        f'<div class="logo-button">{email[0].upper()}</div></a>'
    ) if is_logged_in else ''
    context = {
        "request": request,
        "signup_success": signup_success,
        "message": message,
        "is_logged_in": is_logged_in,
        "logo_button": logo_button,
        "is_logged_in_js": str(is_logged_in).lower(),
    }
    return templates.TemplateResponse("home.html", context)



@app.post("/register")
def register(request: Request, email: str = Form(...), password: str = Form(...), db: SessionLocal = Depends(get_db)):
    # Check if the user already exists
    user = db.query(User).filter(User.email == email).first()
    if user:
        print("User already exists")
        return Response(status_code=409, content="User already exists")
    
    # Register the user
    hashed_password = pwd_context.hash(password)  # Şifreyi bcrypt ile şifrele
    new_user = User(email=email, password=hashed_password)
    db.add(new_user)
    db.commit()
    
    print("Registration successful")
    return Response(status_code=200, content="Registration successful")


def requires_login(function):
    async def wrapper(*args, **kwargs):
        request = next(arg for arg in args if isinstance(arg, Request))
        access_token = request.cookies.get("access_token")

        if not access_token:
            raise HTTPException(status_code=401, detail="Unauthorized")

        # Add your token verification logic here
        # For example, decode and verify the access token

        # If the token is valid, proceed with the wrapped function
        return await function(*args, **kwargs)

    return wrapper


@app.post("/login")
def login(request: Request, data: OAuth2PasswordRequestForm = Depends(), db: SessionLocal = Depends(get_db)):
    email = data.username
    password = data.password

    # Find the user with the given email
    user = db.query(User).filter(User.email == email).first()

    if not user or not pwd_context.verify(password, user.password):
        print("Invalid email or password")
        return templates.TemplateResponse("home.html", {"request": request, "message": "Invalid email or password"})

    # User is authenticated, generate and return the access token
    token = manager.create_access_token(data=dict(sub=email))
    print("Generated access token:", token)  # Print the generated access token

    print("Login successful for user:", email)
    # Set the access token as a cookie
    response = templates.TemplateResponse("index.html", {"request": request, "message": "Login successful"})
    response.set_cookie(key="token", value=token, httponly=True, max_age=3600)

    return response


async def get_current_user(request: Request, db: SessionLocal = Depends(get_db)):
    token = request.cookies.get("token")  # Çerezi (tokeni) al
    print("token :", token)

    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        print("Decoded payload:", payload)  # Decode edilmiş payload'ı kontrol etmek için
        user_id = payload["sub"]
        print("User ID:", user_id)  # Kullanıcı kimliğini kontrol etmek için
        if user_id:
            user = db.query(User).filter(User.email == user_id).first()
            if user:
                return user
    except JWTError as e:
        print("JWTError:", str(e))  # JWT hatasını kontrol etmek için

    raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/index", response_class=HTMLResponse)
def index(request: Request):
    token = request.cookies.get("token")  # Çerezi (tokeni) al
    if token:
        return templates.TemplateResponse("index.html", {"request": request, "token": token})
    else:
        return RedirectResponse(url="/", status_code=302)


@app.get("/analysis", response_class=HTMLResponse)
def analyze(request: Request):
    token = request.cookies.get("token")  # Çerezi (tokeni) al
    if token:
        return templates.TemplateResponse("analysis.html", {"request": request, "token": token})
    else:
        return RedirectResponse(url="/", status_code=302)


@app.get("/model", response_class=HTMLResponse)
def model(request: Request):
    token = request.cookies.get("token")  # Çerezi (tokeni) al
    if token:
        return templates.TemplateResponse("model.html", {"request": request, "token": token})
    else:
        return RedirectResponse(url="/", status_code=302)


@app.get("/preprocessing", response_class=HTMLResponse)
def analyze(request: Request):
    token = request.cookies.get("token")  # Çerezi (tokeni) al
    if token:
        return templates.TemplateResponse("preprocess.html", {"request": request, "token": token})
    else:
        return RedirectResponse(url="/", status_code=302)


# Define a global variable to store the uploaded file
uploaded_file = None

@app.post("/preprocessing", response_class=HTMLResponse)
async def preprocessing(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    print("Access token found:", current_user.email)

    global uploaded_file
    try:
        # Store the uploaded file in a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.file.read())
            uploaded_file = tmp.name
        
        try:
            # First try reading the CSV file as is
            df = pd.read_csv(uploaded_file)
        except (pd.errors.ParserError, UnicodeDecodeError):
            # If CSV read fails, try reading with utf-8 encoding
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except (pd.errors.ParserError, UnicodeDecodeError):
                try:
                    # If CSV read still fails, try reading as XLSX
                    df = pd.read_excel(uploaded_file)
                except (pd.errors.ParserError, UnicodeDecodeError, XLRDError):
                    # If XLSX read fails, try reading with utf-8 encoding
                    try:
                        df = pd.read_excel(uploaded_file, engine='openpyxl', encoding='utf-8')
                    except (pd.errors.ParserError, UnicodeDecodeError):
                        # If utf-8 decoding fails, try reading with a different encoding
                        df = pd.read_excel(uploaded_file, engine='openpyxl', encoding='latin-1')
        
        output = preprocess(df=df)

        if isinstance(output, pd.DataFrame):
            output_file = os.path.join(os.getcwd(), "preprocessed.csv")
            output.to_csv(output_file, index=False)  # Save DataFrame as CSV

            return templates.TemplateResponse("preprocess.html", {"request": request})

        return JSONResponse(content={"message": "Preprocessing completed successfully", "result": output})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)



@app.get("/result_preprocessing", response_class=HTMLResponse)
async def show_result(request: Request):
    try:
        # Here, you can read the CSV file that was saved earlier and return it as a JSON response if needed.
        # This is just an example of how you might do it:
        output_file = os.path.join(os.getcwd(), "preprocessed.csv")
        if os.path.exists(output_file):
            output_df = pd.read_csv(output_file)
            return templates.TemplateResponse('preprocess.html',{"request": request},content=output_df.to_dict(orient="records"))
            
        return JSONResponse(content={"error": "Result file not found"}, status_code=404)

    except Exception as e:
        return f"Error: {str(e)}"



@app.post("/analysis", response_class=HTMLResponse)
async def run_analysis_api(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    print("Access token found:", current_user.email)

    global uploaded_file
    try:
        # Store the uploaded file in a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.file.read())
            uploaded_file = tmp.name
        
        try:
            # First try reading the file as CSV
            df = pd.read_csv(uploaded_file)
        except (pd.errors.ParserError, UnicodeDecodeError):
            # If CSV read fails, try reading with utf-8 encoding
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except (pd.errors.ParserError, UnicodeDecodeError):
                try:
                    # If CSV read still fails, try reading as XLSX
                    df = pd.read_excel(uploaded_file)
                except (pd.errors.ParserError, UnicodeDecodeError, XLRDError):
                    # If XLSX read fails, try reading with utf-8 encoding
                    try:
                        df = pd.read_excel(uploaded_file, engine='openpyxl', encoding='utf-8')
                    except (pd.errors.ParserError, UnicodeDecodeError):
                        # If utf-8 decoding fails, try reading with a different encoding
                        df = pd.read_excel(uploaded_file, engine='openpyxl', encoding='latin-1')
        
        columns = df.columns.tolist()

        return templates.TemplateResponse("analysis.html", {"request": request, "columns": columns})
    
    except Exception as e:
        traceback.print_exc()
        return f"Error occurred while processing the file: {str(e)}"

    

@app.post("/run_analysis", response_class=HTMLResponse)
async def run_analysis(
    request: Request,
    target: str = Form(None),
    current_user: User = Depends(get_current_user)
):
    try:
        global uploaded_file

        file_extension = uploaded_file.filename.split('.')[-1]

        if file_extension == 'csv':
            # Try reading the CSV file using multiple encodings
            read_encodings = ['utf-8', 'latin-1']
            for encoding in read_encodings:
                try:
                    df = pd.read_csv(uploaded_file.file, encoding=encoding)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    pass
        elif file_extension == 'xlsx':
            # Try reading the Excel file using multiple encodings
            read_encodings = ['utf-8', 'latin-1']
            for encoding in read_encodings:
                try:
                    df = pd.read_excel(uploaded_file.file, engine='openpyxl', encoding=encoding)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError, XLRDError):
                    pass
        else:
            return "Unsupported file format"

        df = preprocess(df)

        output = analysis(df=df, target=target)
        pca_dict = {}

        result_dict = calculate_pca(df.select_dtypes(include=['float', 'int']))
        pca_dict = {
            'Cumulative Explained Variance Ratio': result_dict['Cumulative Explained Variance Ratio'],
            'Principal Component': result_dict['Principal Component']
        }

        output['PCA'] = pca_dict
        output = set_to_list(output)
        output_analysis = {"Results": output}

        # Save analysis results to a JSON file (e.g., analysis.json)
        with open("analysis.json", "w") as f:
            json.dump(output_analysis, f)

        # Redirect to the result page
        return RedirectResponse(url="/result_analysis", status_code=302)

    except Exception as e:
        traceback.print_exc()
        return f"Error occurred during analysis: {str(e)}"




@app.get("/result_analysis", response_class=HTMLResponse)
async def show_result(request: Request):
    try:
        with open("analysis.json", "r") as f:
            output_analysis = json.load(f)

        # Render the HTML template
        return templates.TemplateResponse("analysis.html", {"request": request, "result": json.dumps(output_analysis)})
    except Exception as e:
        return f"Error: {str(e)}"
    

@app.post("/model", response_class=HTMLResponse)
def run_model_api(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
    ):
    print("Access token found:", current_user.email)

    global uploaded_file
    try:
        # Store the uploaded file in a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.file.read())
            uploaded_file = tmp.name
        
        try:
            # First try reading the CSV file as is
            df = pd.read_csv(uploaded_file)
        except (pd.errors.ParserError, UnicodeDecodeError):
            # If CSV read fails, try reading with utf-8 encoding
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except (pd.errors.ParserError, UnicodeDecodeError):
                try:
                    # If CSV read still fails, try reading as XLSX
                    df = pd.read_excel(uploaded_file)
                except (pd.errors.ParserError, UnicodeDecodeError, XLRDError):
                    # If XLSX read fails, try reading with utf-8 encoding
                    try:
                        df = pd.read_excel(uploaded_file, engine='openpyxl', encoding='utf-8')
                    except (pd.errors.ParserError, UnicodeDecodeError):
                        # If utf-8 decoding fails, try reading with a different encoding
                        df = pd.read_excel(uploaded_file, engine='openpyxl', encoding='latin-1')
        
        columns = df.columns.tolist()

        return templates.TemplateResponse("model.html", {"request": request, "columns": columns})
    
    except Exception as e:
        traceback.print_exc()
        return f"Error occurred while processing the file: {str(e)}"



@app.post("/run_models", response_class=HTMLResponse)
async def run_models(
    request: Request,
    target: str = Form(None),
    current_user: User = Depends(get_current_user)
):
    print("Access token found:", current_user.email)

    try:
        global uploaded_file

        # Try reading the CSV file using utf-8 encoding
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            # If utf-8 decoding fails, try reading with a different encoding
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        except (UnicodeDecodeError, pd.errors.ParserError):
            # If CSV read fails, try reading as XLSX
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl', encoding='utf-8')
            except (pd.errors.ParserError, UnicodeDecodeError):
                # If utf-8 decoding fails, try reading with a different encoding
                df = pd.read_excel(uploaded_file, engine='openpyxl', encoding='latin-1')

        df = preprocess(df)
        
        if target is None:
            problem_type, params = get_problem_type(df, target=None) # problem_type and params 

            output = create_model(df=df, problem_type=problem_type, params=params)

            # Convert NumPy array to list
            output = output.tolist() if isinstance(output, np.ndarray) else output

            output_file = os.path.join(os.getcwd(), "model.json")

            with open(output_file, "w") as f:
                json.dump(output, f)

            # Redirect to the '/model' endpoint
            return RedirectResponse(url="/result_model", status_code=302)

        else:
            problem_type, params = get_problem_type(df, target=target) # problem_type and params 

            output = create_model(df=df, problem_type=problem_type, params=params)

            # Convert NumPy array to list
            output = output.tolist() if isinstance(output, np.ndarray) else output

            output_file = os.path.join(os.getcwd(), "model.json")

            with open(output_file, "w") as f:
                json.dump(output, f)

            # Redirect to the '/model' endpoint
            return RedirectResponse(url="/result_model", status_code=302)

    
    except Exception as e:
        traceback.print_exc()
        return f"Dataset is not excepted, an error occurred : {str(e)}"


    
    
@app.get("/result_model", response_class=HTMLResponse)
async def show_prediction(request: Request):
    try:
        with open("model.json", "r") as f:
            output_model = json.load(f)

        # Render the HTML template
        return templates.TemplateResponse("model.html", {"request": request, "result": json.dumps(output_model)})
    except Exception as e:
        return f"Error: {str(e)}"


uvicorn.run(app, host="127.0.0.1", port=5050)

