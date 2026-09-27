import os

os.environ["HF_HUB_OFFLINE"] = "1"

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, GenerationConfig

app = FastAPI()

generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct"
)

generation_config = GenerationConfig(
    max_new_tokens=200
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):

    messages = [
        {
            "role": "user",
            "content": request.message
        }
    ]

    result = generator(
        messages,
        generation_config=generation_config,
        clean_up_tokenization_spaces=False
    )

    response = result[0]["generated_text"]

    return {
        "response": response
    }
