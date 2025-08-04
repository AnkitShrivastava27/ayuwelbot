from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os, re
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI()

# CORS for Flutter or Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, set your frontend domain here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace Azure with Hugging Face
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    raise ValueError("Set HUGGINGFACEHUB_API_TOKEN in env variables")

client = InferenceClient(token=api_token)
model_name = "HuggingFaceH4/zephyr-7b-beta"  # Free and supports text generation

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_input = request.question.strip()
        if not user_input:
            return {"response": "Please enter a valid question."}

        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"

        result = client.text_generation(
            prompt=prompt,
            model=model_name,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
        )

        output = result.strip()
        output = re.sub(r"<.*?>", "", output)
        output = re.sub(r"\s+", " ", output)

        return {"response": output}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@app.get("/")
def root():
    return {"message": "Free Hugging Face Chat API is running."}
