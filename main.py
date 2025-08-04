import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# Azure OpenAI setup
llm_model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

# Prompt template
chat_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
{chat_history}
User question: {question}
Answer:
""")

# Memory setup (global memory for demo; in production use session IDs)
memory = ConversationSummaryBufferMemory(
    llm=llm_model,
    max_token_limit=1000,
    return_messages=True,
    input_key="question",
    output_key="text",
    memory_key="chat_history",
)

# LangChain chain
llm_chain = LLMChain(
    llm=llm_model,
    prompt=chat_prompt,
    memory=memory,
)

# Define input schema
class ChatRequest(BaseModel):
    question: str

# API endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_input = request.question.strip()
        if not user_input:
            return {"response": "Please enter a valid question."}

        result = llm_chain.invoke({"question": user_input})
        response = result["text"]

        return {
            "response": response,
            "chat_history": memory.chat_memory.messages,  # optional, returns full memory
        }

    except Exception as e:
        return {"error": str(e)}

# Optional health check endpoint
@app.get("/")
def read_root():
    return {"message": "Azure GPT Chat API is running"}
