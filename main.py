import os

os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import pipeline, GenerationConfig

generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct"
)

generation_config = GenerationConfig(
    max_new_tokens=200
)

while True:
    prompt = input("You > ")

    if prompt.lower() == "exit":
        break


    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    result = generator(
        messages,
        generation_config=generation_config,
        clean_up_tokenization_spaces=False
    )

    response = result[0]["generated_text"][-1]["content"]

    print("AI >", response)
