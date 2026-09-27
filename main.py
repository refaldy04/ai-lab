import os

os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct"
)

messages = [
    {
        "role": "user",
        "content": "Explain what TCP is in one sentence."
    }
]

result = generator(
    messages,
    max_new_tokens=100
)

print(result[0]["generated_text"])