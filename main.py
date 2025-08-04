from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
import re
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI()

# Allow Flutter and other frontends to access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Hugging Face API token
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(api_key=api_key)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message

    # System + User message for Zephyr-style chat model
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional medical assistant. Only answer medical-related questions. "
                "Do not repeat the user's question. Keep your response brief—3–4 lines or under 60 words. "
                "Avoid storytelling or follow-up prompts. If the question is not medical, say: "
                "'I'm only able to assist with medical-related questions.'"
            )
        },
        {"role": "user", "content": user_input}
    ]

    try:
        response = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=messages,
            max_tokens=120,
            temperature=0.5,
            stream=False
        )

        final_answer = response.choices[0].message.content

        # Remove unnecessary junk or prompts
        clean = re.sub(r"<.*?>|\[/?[A-Z]+\]", "", final_answer)  # Remove <tags> and [INST] stuff
        clean = re.sub(r"(?i)(you asked|user:|question:).*", "", clean)  # Avoid echoed input
        clean = re.sub(r"\s+", " ", clean).strip()

        # Trim to first 3–4 sentences only
        sentences = re.split(r'(?<=[.!?]) +', clean)
        short_reply = " ".join(sentences[:4])

        disclaimer = " As I'm a general chatbot model, please consult a doctor for serious conditions."

        return {"response": short_reply + disclaimer}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
