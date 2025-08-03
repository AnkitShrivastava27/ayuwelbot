from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os, re
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI()

# Allow all CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your Hugging Face token as env variable or directly here
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(api_key=api_key)

# Define expected request body
class ChatRequest(BaseModel):
    message: str

# POST endpoint to handle chat requests
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
        response = client.text_generation(
            model="google/flan-t5-base",  # ✅ Free & Supported
            prompt=prompt,
            max_new_tokens=100
        )

        output = response.strip()
        output = re.sub(r"<.*?>", "", output)
        output = re.sub(r"\s+", " ", output)
        disclaimer = " As I'm an AI model, please consult a licensed doctor for serious health concerns."

        return {"response": output + disclaimer}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
