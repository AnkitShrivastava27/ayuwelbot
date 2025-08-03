from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os, re
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(token=api_token)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message.strip()
    if not user_input:
        return {"response": "Please enter a valid question."}

    prompt = (
        "You are a professional medical assistant. Answer clearly and briefly.\n"
        f"Question: {user_input}\nAnswer:"
    )

    try:
        # Use prompt as positional argument and the correct model ID
        result = client.text_generation(
            prompt,
            model="prithivMLmods/Deepthink-Llama-3-8B-Preview",
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
        )

        output = result.strip()
        output = re.sub(r"<.*?>", "", output)
        output = re.sub(r"\s+", " ", output)
        disclaimer = " As I'm an AI model, please consult a licensed doctor for serious health concerns."

        return {"response": output + disclaimer}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
