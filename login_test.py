from fastapi import FastAPI, Request, Depends, Form, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
# from sqlalchemy.ext.declarative import declarative_base
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi_login import LoginManager
from passlib.context import CryptContext

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

# Template dosyalarını yükleme
templates = Jinja2Templates(directory="template")

# Statik dosyaları sunma
app.mount("/dist", StaticFiles(directory="dist"), name="dist")
app.mount("/plugins", StaticFiles(directory="plugins"), name="plugins")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, signup_success: bool = False, message: str = "", email: str = ""):
    # Determine if the user is logged in
    is_logged_in = bool(email)
    
    # Generate the logo button if the user is logged in
    logo_button = f'<a class="nav-link" href="#" style="display: block;"><div class="logo-button">{email[0].upper()}</div></a>' if is_logged_in else ''
    
    # Render the template with the appropriate context variables
    context = {
        "request": request,
        "signup_success": signup_success,
        "message": message,
        "is_logged_in": is_logged_in,
        "logo_button": logo_button,
        "is_logged_in_js": str(is_logged_in).lower(),  # Convert to string and lowercase
    }

    return templates.TemplateResponse("index.html", context)


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



@app.post("/login")
def login(request: Request, data: OAuth2PasswordRequestForm = Depends(), db: SessionLocal = Depends(get_db)):
    email = data.username
    password = data.password

    # Find the user with the given email
    user = db.query(User).filter(User.email == email).first()

    if not user or not pwd_context.verify(password, user.password):
        print("Invalid email or password")
        return templates.TemplateResponse("index.html", {"request": request, "message": "Invalid email or password"})

    # User is authenticated, generate and return the access token
    token = manager.create_access_token(data=dict(sub=email))

    print("Login successful for user:", email)
    # Set the access token as a cookie and redirect to the homepage
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(key="access_token", value=token)
    return response




@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    return RedirectResponse(url="/", status_code=exc.status_code)


@app.get("/content")
def get_content(request: Request, token: str = Depends(manager)):
    return templates.TemplateResponse("content.html", {"request": request})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5050)
