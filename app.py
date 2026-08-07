from pypdf import PdfReader
from pathlib import Path
from preprocess import preprocess_pdf
from embedder import generate_embeddings
from model import embedding_model
from faiss_store import build_index, search
from prompt import build_pormpt
from llm import generate_answer

def main():
    path = Path("data/documents/ML_u1.pdf")
    chunks = preprocess_pdf(path, chunk_size=100)

    chunks = generate_embeddings(chunks)
    index = build_index(chunks)

    query = input("Ask a question: ")
    query_embedding = embedding_model.encode(query)
    retrieved_chunks = search(index, query_embedding, chunks, k = 20)
    prompt = build_pormpt(query, retrieved_chunks)
    answer = generate_answer(prompt)

    print("Answer")
    print(answer)
    print()

if __name__ == "__main__":
    main()