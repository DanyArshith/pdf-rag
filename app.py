from pypdf import PdfReader
from pathlib import Path
from preprocess import preprocess_pdf
from embedder import generate_embeddings
from model import embedding_model
from faiss_store import build_index, search
from prompt import build_prompt
from llm import generate_answer

def main():
    path = Path("data/documents/ML_u1.pdf")
    chunks = preprocess_pdf(path, chunk_size=300, overlap = 50)
    chunks = generate_embeddings(chunks)
    index = build_index(chunks)

    while True:
        query = input("Ask a question: ").strip()
        if query.lower() == "exit":
            break

        query_embedding = embedding_model.encode(query)
        retrieved_chunks = search(index, query_embedding, chunks, k = 8)
        prompt = build_prompt(query, retrieved_chunks)

        print("\n\nAnswer:\n")
        answer = generate_answer(prompt)
        print()


if __name__ == "__main__":
    main()