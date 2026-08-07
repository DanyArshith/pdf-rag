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
        ]
     )
     return response["message"]["content"]