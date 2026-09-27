import os

os.environ["HF_HUB_OFFLINE"] = "1"

from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, GenerationConfig

app = FastAPI()

# Disimpan di RAM selama proses server berjalan.
conversations: dict[str, list[dict[str, str]]] = {}
chat_lock = Lock()

generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct"
)

generation_config = GenerationConfig(
    max_new_tokens=200
)

class ChatRequest(BaseModel):
    conversation_id: str
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    # Untuk demo lokal: proses chat satu per satu agar riwayat
    # tidak bertabrakan saat dua request datang bersamaan.
    with chat_lock:
        history = conversations.get(
            request.conversation_id,
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                }
            ],
        )

        # Buat daftar baru; simpan setelah generasi berhasil.
        messages = history + [
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

        answer = result[0]["generated_text"][-1]["content"]

        conversations[request.conversation_id] = messages + [
            {"role": "assistant", "content": answer}
        ]


    return {
        "conversation_id": request.conversation_id,
        "response": answer,
    }
