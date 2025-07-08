from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
import re
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (Flutter access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Flutter frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                "You are a professional medical assistant. Only respond to medical-related questions. "
                "If the question is not medical, simply respond: 'I'm only able to assist with medical-related questions.'"
            )
        },
        {"role": "user", "content": user_input}
    ]

    try:
        response = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=messages,
            stream=False
        )
        final_answer = response.choices[0].message.content
        clean_answer = re.sub(r"<think>.*?</think>", "", final_answer, flags=re.DOTALL).strip()
        return {"response": clean_answer}
    except Exception as e:
        return {"error": str(e)}
