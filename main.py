from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import re

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FLAN-T5 pipeline
flan_pipe = pipeline("text2text-generation", model="google/flan-t5-base")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message.strip()
    if not user_input:
        return {"response": "Please enter a valid medical question."}

    prompt = (
        "You are a professional medical assistant. Answer clearly and briefly.\n"
        f"Question: {user_input}\nAnswer:"
    )

    try:
        result = flan_pipe(prompt, max_new_tokens=100)[0]["generated_text"]
        result = re.sub(r"<.*?>", "", result)
        result = re.sub(r"\s+", " ", result)
        disclaimer = " As I'm an AI model, always consult a licensed doctor for serious conditions."

        return {"response": result + disclaimer}

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}
