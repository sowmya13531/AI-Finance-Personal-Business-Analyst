from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=100,
    temperature=0.5,
    do_sample=True,
    return_full_text=False
)

def ask_llm(prompt):

    result = generator(prompt)

    text = result[0]["generated_text"]

    return text.strip()