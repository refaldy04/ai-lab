from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    local_files_only=True,
)

messages = [
    {"role": "user", "content": "Explain what a neural network is."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))