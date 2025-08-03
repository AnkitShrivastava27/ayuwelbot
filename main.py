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
    allow_headers=["*"]
)

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(api_key=api_key)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message

    prompt = (
        "You are a professional medical assistant. Only answer medical-related questions. "
        "Respond in 3–4 lines (max ~60 words). No storytelling. "
        "If not medical, say: 'I'm only able to assist with medical-related questions.'\n\n"
        f"Question: {user_input}\nAnswer:"
    )

    try:
        response = client.text_generation(
            model="tiiuae/falcon-rw-1b",  
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.5,
            do_sample=True
        )

        answer = response.strip()
        answer = re.sub(r"<.*?>", "", answer)
        answer = re.sub(r"\s+", " ", answer).strip()
        disclaimer = " As I'm a general chatbot model, please consult a doctor for serious conditions."
        return {"response": answer + disclaimer}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
