from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os, re
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Flutter domain if needed
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

    # Mistral-style prompt using [INST] format
    system_prompt = (
        "You are a helpful and concise medical assistant. "
        "Answer only medical-related questions in 3–4 lines (about 60 words max). "
        "Avoid long explanations. If the question is not medical, say: "
        "'I'm only able to assist with medical-related questions.'"
    )

    prompt = f"[INST] <<SYS>> {system_prompt} <</SYS>> {user_input} [/INST]"

    try:
        response = client.text_generation(
            model="mistralai/Mistral-7B-Instruct-v0.1",  # ✅ Free and accurate
            prompt=prompt,
            max_new_tokens=120,
            temperature=0.4,
            do_sample=True
        )

        answer = response.strip()
        answer = re.sub(r"<.*?>", "", answer)
        answer = re.sub(r"\s+", " ", answer).strip()
        disclaimer = " As I'm a general chatbot model, please consult a doctor for serious conditions."

        return {"response": answer + disclaimer}
    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
