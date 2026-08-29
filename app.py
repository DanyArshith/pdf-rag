from pypdf import PdfReader
from pathlib import Path
from preprocess import preprocess_pdf
from embedder import generate_embeddings
from model import embedding_model
from faiss_store import (
    build_index,
    save_index,
    load_index,
    search,
    debug_search
)
from prompt import build_prompt
from llm import generate_answer
from evaluation import evaluation_data, evaluate_all

def main():
    path = Path("data/documents/GEN_AI.pdf")

    chunks = preprocess_pdf(path, chunk_size=400, overlap = 50)
    chunks = generate_embeddings(chunks)
    index = build_index(chunks)
    
    query = "What are the two competing networks in a GAN?"
    query_embedding = embedding_model.encode(query)

    result = debug_search(
        index,
        query_embedding,
        chunks,
        k=20
    )

    for rank, item in enumerate(result, start=1):
        chunk = item["chunk"]

        print(f"\nRank {rank}")
        print(f"Distance: {item['distance']:.4f}")
        print(f"Page: {chunk['page']}")
        print(f"Text: {chunk['text']}")


if __name__ == "__main__":
    main()