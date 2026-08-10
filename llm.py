from ollama import chat
MODEL = "llama3.2:3b"

def generate_answer(prompt):
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream = True
    )

    answer = ""
    for chunk in response:
        text = chunk["message"]["content"]
        print(text, end="", flush=True)
        answer += text

    return answer

    