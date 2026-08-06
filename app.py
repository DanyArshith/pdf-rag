from pypdf import PdfReader
from pathlib import Path
from preprocess import preprocess_pdf
from embedder import generate_embeddings
from vector_store import search
from model import embedding_model
from faiss_store import build_index

def main():
    path = Path("data/documents/ML_u1.pdf")
    chunks = preprocess_pdf(path, chunk_size=100)

    chunks = generate_embeddings(chunks)
    index = build_index(chunks)
    # query = "What is Machine Learning"
    # query_embedding = embedding_model.encode(query)

    # similar_chunks = search(query_embedding, chunks, k = 10)
    # print(similar_chunks)


if __name__ == "__main__":
    main()