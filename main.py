from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os, re
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(api_key=api_key)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional medical assistant. Only answer medical questions. "
                "Keep your response concise—3–4 lines or 60 words max. "
                "If the question is not medical, say: 'I'm only able to assist with medical-related questions.'"
            )
        },
        {"role": "user", "content": user_input}
    ]

    try:
        resp = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=messages,
            stream=False,
            max_tokens=120
        )
        output = resp.choices[0].message.content.strip()
        output = re.sub(r"<.*?>", "", output)
        output = re.sub(r"\s+", " ", output)
        disclaimer = " As I'm a general chatbot model, please consult a doctor for serious conditions."
        return {"response": output + disclaimer}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
