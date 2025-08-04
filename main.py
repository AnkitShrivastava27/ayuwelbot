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
if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not set.")

client = InferenceClient(token=api_token)
model_name = "prithivMLmods/Deepthink-Llama-3-8B-Preview"

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message.strip()
    if not user_input:
        return {"response": "Please enter a valid question."}

    try:
        result = client.chat_conversational(
            model=model_name,
            messages=[{"role": "user", "content": user_input}],
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
        )

        output = result.strip() if isinstance(result, str) else result.get("generated_text", "")
        output = re.sub(r"<.*?>", "", output)
        output = re.sub(r"\s+", " ", output)

        disclaimer = " As I'm an AI model, please consult a licensed doctor for serious health concerns."
        return {"response": output + disclaimer}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
