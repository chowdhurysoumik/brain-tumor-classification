from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from database import engine, Base
import models
from sqlalchemy.orm import Session
from fastapi import Depends
from database import SessionLocal
from auth import register_user, login_user

from model_service import predict_image
from lime_service import explain_image

Base.metadata.create_all(bind=engine)
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()

app = FastAPI(
    title="Brain Tumor AI API",
    description="Brain MRI classification API",
    version="1.0"
)

@app.post("/register")
def register(
    name: str,
    email: str,
    password: str,
    db: Session = Depends(get_db)
):

    user = register_user(
        db,
        name,
        email,
        password
    )

    if user is None:
        return {
            "success": False,
            "message": "Email already registered"
        }

    return {
        "success": True,
        "message": "Registration successful"
    }
@app.post("/login")
def login(
    email: str,
    password: str,
    db: Session = Depends(get_db)
):

    user = login_user(
        db,
        email,
        password
    )

    if user is None:
        return {
            "success": False,
            "message": "Invalid email or password"
        }

    return {
        "success": True,
        "message": "Login successful",
        "name": user.name,
        "email": user.email
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Brain Tumor AI API is running"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    result = predict_image(image)

    return result

@app.post("/explain")
async def explain(
    file: UploadFile = File(...)
):
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    lime_image_array = explain_image(image)

    image_buffer = io.BytesIO()

    Image.fromarray(
        lime_image_array
    ).save(
        image_buffer,
        format="PNG"
    )

    image_buffer.seek(0)

    return StreamingResponse(
        image_buffer,
        media_type="image/png"
    )