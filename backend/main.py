from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel,Field

app = FastAPI(title = "SpamShield AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://spam-shield-ai-pink.vercel.app"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR/"spam_model.joblib"
model = joblib.load(MODEL_PATH)

class MessageRequest(BaseModel):
    message:str = Field(min_length=1,max_length=5000)

@app.get("/")
def home():
    return {"message":"SpamShield AI is running"}

@app.post("/api/predict")
def predict_spam(request:MessageRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="message cannot be empty!"
        )
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = model.named_steps["classifier"].classes_
    probability_map = dict(zip(classes,probabilities))

    confidence = float(probability_map[prediction])

    return {
        "message": message,
        "prediction": "Spam" if prediction == "spam" else "Not Spam",
        "confidence": round(confidence * 100, 2)
    }




