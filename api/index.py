from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="PharmaChatbot API")

# ── CORS — allow your Android app and any frontend ───────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# ── System prompt — focuses the model on medical/pharma topics ─
SYSTEM_PROMPT = """You are PharmBot, a knowledgeable pharmaceutical and medical assistant.
You help users with:
- Medicine information (uses, dosage, side effects, interactions)
- General health and wellness queries
- Drug identification and descriptions
- Prescription guidance (remind users to consult a doctor)
- Symptom information (always advise professional consultation)

Always be clear, accurate, and responsible. When in doubt, always recommend
the user consult a licensed healthcare professional or pharmacist.
Do not diagnose diseases. Do not prescribe medication.
Keep responses concise and easy to understand."""

# ── Chat endpoint ─────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY not configured.")

    payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": request.message}
        ],
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="NVIDIA API timed out.")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"NVIDIA API error: {e.response.text}")

    data = response.json()
    reply = data["choices"][0]["message"]["content"]
    return ChatResponse(reply=reply)

# ── Health check ──────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "meta/llama-3.1-8b-instruct"}

# ── Root ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "PharmaChatbot API is running. POST to /api/chat"}
