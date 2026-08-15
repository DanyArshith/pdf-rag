def build_prompt(question, chunks):
    context = ""
    for chunk in chunks:

        context += f"Page[{chunk["page"]}]\n"
        context += f"{chunk["text"]}" +"\n\n"

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the context below.

If the answer cannot be found in the context, say:
"I couldn't find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt