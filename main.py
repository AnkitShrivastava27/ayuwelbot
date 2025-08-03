from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os, re
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(api_key=api_key)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message

    # Mistral-style system + instruction prompt
    system_prompt = (
        "You are a professional medical assistant. Only respond to medical questions. "
        "Keep it concise: 3–4 lines or ~50–60 words. No storytelling. "
        "If not medical, respond: 'I'm only able to assist with medical-related questions.'"
    )

    prompt = f"[INST] <<SYS>> {system_prompt} <</SYS>> {user_input} [/INST]"

    try:
        response = client.text_generation(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            prompt=prompt,
            max_new_tokens=120,
            temperature=0.3,
            do_sample=True
        )
        answer = response.strip()
        answer = re.sub(r"<.*?>", "", answer)
        answer = re.sub(r"\s+", " ", answer).strip()
        disclaimer = " As I'm a general chatbot model, in severe cases please consult a doctor."
        return {"response": answer + disclaimer}
    except Exception as e:
        return {"error": str(e)}
