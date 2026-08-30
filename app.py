from pathlib import Path
from preprocess import preprocess_pdf
from embedder import generate_embeddings
from faiss_store import build_index
from evaluation import evaluation_data, evaluate_all


def main():
    path = Path("data/documents/GEN_AI.pdf")

    chunks = preprocess_pdf(
        path,
        chunk_size=400,
        overlap=50
    )

    chunks = generate_embeddings(chunks)
    index = build_index(chunks)

    k_values = [1, 3, 5, 10, 20]

    for k in k_values:

        recall = evaluate_all(
            index,
            chunks,
            evaluation_data,
            k=k,
            rerank_top_n=20
        )

        print(f"\nRecall@{k}: {recall:.2f}%")


if __name__ == "__main__":
    main()