from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def root():
    return {"status": "Pharma chatbot backend running (Gemini)"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured"
        )

    try:
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-pro")

response = model.generate_content(
    f"You are a helpful pharmacy assistant.\nUser: {req.message}",
    generation_config={
        "temperature": 0.3,
        "max_output_tokens": 150
    }
)


        return {"reply": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

