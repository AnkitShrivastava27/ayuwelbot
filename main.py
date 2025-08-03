from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os, re
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI()

# Enable CORS for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use your Flutter domain if hosted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API setup
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(api_key=api_key)

# Request schema
class ChatRequest(BaseModel):
    message: str

# Endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message

    # Clear instruction
    system_prompt = (
        "You are a professional medical assistant. Only respond to medical questions. "
        "Keep it concise: 3–4 lines or ~50–60 words. No storytelling. "
        "If not medical, respond: 'I'm only able to assist with medical-related questions.'"
    )

    # Use a free, compatible HF-hosted model
    prompt = f"{system_prompt}\nPatient: {user_input}\nAssistant:"

    try:
        response = client.text_generation(
            model="tiiuae/falcon-rw-1b",  # ✅ Known to work with HF InferenceClient
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.5,
        )
        answer = response.strip()
        answer = re.sub(r"<.*?>", "", answer)
        answer = re.sub(r"\s+", " ", answer).strip()

        disclaimer = " As I'm a general chatbot model, in severe cases please consult a doctor."
        return {"response": answer + disclaimer}
    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
