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


@app.post("/register")
def register(email: str = Form(...), password: str = Form(...), db: SessionLocal = Depends(get_db)):
    # Check if the user already exists
    user = db.query(User).filter(User.email == email).first()
    if user:
        return RedirectResponse(url="/", status_code=302)
    
    # Register the user
    new_user = User(email=email, password=password)
    db.add(new_user)
    db.commit()
    
    return RedirectResponse(url="/", status_code=302)

# Create a password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@app.post("/login")
def login(data: OAuth2PasswordRequestForm = Depends(), db: SessionLocal = Depends(get_db)):
    email = data.email
    password = data.password
    
    # Find the user with the given email
    user = db.query(User).filter(User.email == email).first()
    if not user or not pwd_context.verify(password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # User is authenticated, generate and return the access token
    token = manager.create_access_token(data=dict(sub=email))
    
    # Redirect to the homepage with a message
    response = Response(headers={"Location": "/"}, status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=token)
    return response

@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    return RedirectResponse(url="/", status_code=exc.status_code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5050)
