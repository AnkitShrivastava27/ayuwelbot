from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
import re
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI()

# Enable CORS (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use your Flutter domain if hosted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get HF API key from environment variable
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(api_key=api_key)

# Request model
class ChatRequest(BaseModel):
    message: str

# POST endpoint for chat
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message

    # Clear instruction prompt
    prompt = (
        "You are a professional medical assistant. Only answer medical-related questions. "
        "Respond in 3–4 lines (under 60 words). "
        "If the question is not medical, say: 'I'm only able to assist with medical-related questions.'\n\n"
        f"Question: {user_input}\nAnswer:"
    )

    try:
        response = client.text_generation(
            model="tiiuae/falcon-rw-1b",  # ✅ Free & supported
            prompt=prompt,
            max_new_tokens=120,
            temperature=0.5,
            do_sample=True
        )

        # Clean the model output
        answer = response.strip()
        answer = re.sub(r"<.*?>", "", answer)
        answer = re.sub(r"\s+", " ", answer).strip()

        # Add safety disclaimer
        disclaimer = " As I'm a general chatbot model, please consult a doctor for serious conditions."

        return {"response": answer + disclaimer}

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
